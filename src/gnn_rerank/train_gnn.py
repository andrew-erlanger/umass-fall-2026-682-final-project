"""Train passage GATs; choose checkpoints using validation data only."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from . import eval as ev
from .gnn import FEATURE_VERSION, GNNReranker, ModelConfig, PassageGAT
from .graph import GraphConfig, GraphDataset, build_graph, dataset_hash


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32  # Questions/graphs per batch, not individual passages.
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 682
    log_every: int = 50

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.log_every) < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid training configuration")


def _write_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def train_one_epoch(model, loader, optimizer, device, *, log_every=50):
    """Fit all training batches once; return mean loss over passage nodes."""
    model.train()
    total_loss, nodes = 0.0, 0
    for step, batch in enumerate(loader, 1):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(batch.x, batch.edge_index), batch.y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite GNN loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
        nodes += batch.num_nodes
        if step == 1 or step % log_every == 0 or step == len(loader):
            print(f"batch {step}/{len(loader)} loss={total_loss / nodes:.4f}")
    return total_loss / nodes


def train(train_rows, val_rows, train_emb, val_emb, output_dir, *, model_config=None, graph_config=None,
          config=None, device=None):
    """Fit train only, select best epoch on val, and reuse matching completed runs.

    Loss is unweighted binary cross-entropy averaged over passage nodes. All
    distractors are retained. A nonempty interrupted run is not resumed: use a
    new output directory. Optimizer state is not saved. No dev data is accepted.
    """
    config = config or TrainConfig()
    graph_config = graph_config or GraphConfig()
    model_config = model_config or ModelConfig(train_emb.questions.shape[1] + 1)
    if not len(train_rows) or not len(val_rows):
        raise ValueError("Train and validation splits must be nonempty")
    if train_emb.model != val_emb.model or train_emb.questions.shape[1] != val_emb.questions.shape[1]:
        raise ValueError("Train/val must use the same embedding encoder and dimensions")
    # Hash rows (including labels) to prevent silently reusing a checkpoint for
    # a different experiment. Embeddings are independently checked per graph.
    manifest = {
        "feature_version": FEATURE_VERSION, "encoder": train_emb.model,
        "model": asdict(model_config), "graph": asdict(graph_config), "training": asdict(config),
        "train_hash": dataset_hash(train_rows), "val_hash": dataset_hash(val_rows),
    }
    output_dir = Path(output_dir)
    if (output_dir / "complete.json").exists():
        if json.loads((output_dir / "experiment.json").read_text()) != manifest:
            raise ValueError("Completed run used different data/settings; choose a new RUN_ID")
        if not (output_dir / "best.pt").exists():
            raise FileNotFoundError("Completed run is missing best.pt")
        print(f"Reusing completed run: {output_dir}")
        return json.loads((output_dir / "history.json").read_text())
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Interrupted/nonempty run at {output_dir}; choose a new RUN_ID")

    seed_everything(config.seed)
    model = PassageGAT(model_config)
    val_scorer = GNNReranker(model, val_emb, graph_config, device=device)
    dataset = GraphDataset(train_rows, train_emb, graph_config, with_labels=True)
    # Materialize one graph before creating files, to catch input mismatches.
    build_graph(val_rows[0], val_emb, graph_config)
    dataset[0]
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "experiment.json", manifest)
    history, best = [], (-1.0, -1.0)

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        train_loss = train_one_epoch(model, loader, optimizer, val_scorer.device, log_every=config.log_every)
        # evaluate() sets eval mode, disabling dropout. Next epoch restores train.
        metrics = ev.evaluate(val_scorer, val_rows)
        history.append({"epoch": epoch, "train_loss": train_loss, "val": metrics})
        candidate = (metrics["all_gold@5"], metrics["recall@5"])
        if candidate > best:
            best = candidate
            val_scorer.save(output_dir / "best.pt")
        _write_json(output_dir / "history.json", history)
        print(f"validation all_gold@5={candidate[0]:.3f} recall@5={candidate[1]:.3f}")

    # Written last: a finished marker cannot refer to a half-saved checkpoint.
    _write_json(output_dir / "complete.json", {"best_all_gold@5": best[0], "best_recall@5": best[1]})
    return history
