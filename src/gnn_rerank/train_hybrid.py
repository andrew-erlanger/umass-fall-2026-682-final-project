"""Train only the hybrid GAT, reusing the standalone GNN's epoch routine."""

from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from . import eval as ev
from .gnn import ModelConfig, PassageGAT
from .graph import GraphConfig, dataset_hash
from .hybrid import FEATURE_VERSION, HybridDataset, HybridReranker, build_hybrid_graph
from .train_gnn import TrainConfig, seed_everything, train_one_epoch


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def train(train_rows, val_rows, train_emb, val_emb, train_scores, val_scores, output_dir, *,
          model_config=None, graph_config=None, config=None, device=None):
    """Fit a stage-wise hybrid: frozen CE predictions, frozen BGE, learned GAT.

    No dev data or CE parameters are accepted by this training function. Source
    and score fingerprints prevent reuse after changing the upstream model.
    """
    config = config or TrainConfig()
    graph_config = graph_config or GraphConfig()
    model_config = model_config or ModelConfig(train_emb.questions.shape[1] + 2)
    if not len(train_rows) or not len(val_rows):
        raise ValueError('Train and validation must be nonempty')
    if train_emb.model != val_emb.model or train_emb.questions.shape[1] != val_emb.questions.shape[1]:
        raise ValueError('Train/val embeddings must use the same encoder and dimensions')
    if train_scores.source != val_scores.source:
        raise ValueError('Train/val scores must come from the same frozen cross-encoder')
    manifest = {
        'feature_version': FEATURE_VERSION, 'encoder': train_emb.model,
        'model': asdict(model_config), 'graph': asdict(graph_config), 'training': asdict(config),
        'train_hash': dataset_hash(train_rows), 'val_hash': dataset_hash(val_rows),
        'train_scores': train_scores.signature(), 'val_scores': val_scores.signature(),
    }
    output_dir = Path(output_dir)
    if (output_dir / 'complete.json').exists():
        if json.loads((output_dir / 'experiment.json').read_text()) != manifest:
            raise ValueError('Hybrid run has different settings/data/CE scores; choose a new RUN_ID')
        if not (output_dir / 'best.pt').exists():
            raise FileNotFoundError('Completed hybrid run is missing best.pt')
        print('Reusing completed hybrid:', output_dir)
        return json.loads((output_dir / 'history.json').read_text())
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Interrupted/nonempty hybrid run; choose a new RUN_ID')

    seed_everything(config.seed)
    model = PassageGAT(model_config)
    validation = HybridReranker(model, val_emb, val_scores, graph_config, device=device)
    dataset = HybridDataset(train_rows, train_emb, train_scores, graph_config)
    dataset[0]
    build_hybrid_graph(val_rows[0], val_emb, val_scores, graph_config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / 'experiment.json', manifest)
    history, best = [], (-1., -1.)
    for epoch in range(1, config.epochs + 1):
        print(f'Hybrid epoch {epoch}/{config.epochs}')
        loss = train_one_epoch(model, loader, optimizer, validation.device, log_every=config.log_every)
        metrics = ev.evaluate(validation, val_rows)
        history.append({'epoch': epoch, 'train_loss': loss, 'val': metrics})
        candidate = (metrics['all_gold@5'], metrics['recall@5'])
        if candidate > best:
            best = candidate
            validation.save(output_dir / 'best.pt')
        _write_json(output_dir / 'history.json', history)
        print(f'validation all_gold@5={candidate[0]:.3f} recall@5={candidate[1]:.3f}')
    _write_json(output_dir / 'complete.json', {'best_all_gold@5': best[0], 'best_recall@5': best[1]})
    return history
