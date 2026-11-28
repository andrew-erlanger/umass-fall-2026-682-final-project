"""Text-pair cross-encoder training and the shared reranker adapter.

No embedding cache is used. Labels are used only by PairDataset during training;
score() reads question/title/text, never supporting labels. Heavy dependencies
are imported lazily so pair preparation can also be inspected without a GPU.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import eval as ev

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def passage_text(p: dict) -> str:
    return f"{p['title']}: {p['text']}"


class PairDataset:
    """Index question/passage pairs without duplicating the entire text corpus."""

    def __init__(self, rows):
        from bisect import bisect_right

        self.rows = rows
        self._bisect = bisect_right
        self.offsets = [0]
        for row in rows:
            self.offsets.append(self.offsets[-1] + len(row["passages"]))

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        i = self._bisect(self.offsets, index) - 1
        row = self.rows[i]
        p = row["passages"][index - self.offsets[i]]
        return {"question": row["question"], "passage": passage_text(p), "label": float(p["is_supporting"])}


class CrossEncoderReranker:
    """Return raw logits aligned with row/passage order, in bounded pair batches."""

    name = "cross_encoder"

    def __init__(self, model=DEFAULT_MODEL, *, batch_size=32, max_length=None, device=None, name=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Loading a saved checkpoint restores its truncation length unless overridden.
        settings = Path(model) / "reranker_config.json"
        saved = json.loads(settings.read_text()) if settings.exists() else {}
        self.max_length = max_length if max_length is not None else saved.get("max_length", 512)
        if self.max_length < 4:
            raise ValueError("max_length must be at least 4")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.name = name or self.name
        self.tokenizer = AutoTokenizer.from_pretrained(str(model))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model)).to(self.device)
        if self.model.config.num_labels != 1:
            raise ValueError("Cross-encoder must have a single output logit per pair")

    def _tokenize(self, questions, passages):
        return self.tokenizer(
            questions, passages, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)

    def score(self, rows):
        import torch

        out = [[] for _ in rows]
        pending, owners = [], []

        def flush():
            if not pending:
                return
            features = self._tokenize([p[0] for p in pending], [p[1] for p in pending])
            with torch.inference_mode():
                logits = self.model(**features).logits.reshape(-1).float().cpu().tolist()
            for owner, value in zip(owners, logits, strict=True):
                out[owner].append(value)
            pending.clear()
            owners.clear()

        self.model.eval()
        for i, row in enumerate(rows):
            for p in row["passages"]:
                pending.append((row["question"], passage_text(p)))
                owners.append(i)
                if len(pending) == self.batch_size:
                    flush()
        flush()
        return out

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "reranker_config.json").write_text(json.dumps({"max_length": self.max_length}, indent=2))


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 512
    seed: int = 682
    log_every: int = 50


def train(train_rows, val_rows, output_dir, *, model=DEFAULT_MODEL, config=None, device=None):
    """Fine-tune with unweighted binary cross-entropy on every candidate pair.

    Select the best epoch by validation all_gold@5, then recall@5 on ties. Never
    reads dev. Refuses a nonempty output directory to avoid reusing stale weights.
    Checkpoints contain model/tokenizer, not optimizer state (no training resume).
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import set_seed

    config = config or TrainConfig()
    if config.epochs < 1 or config.batch_size < 1 or config.learning_rate <= 0 or config.log_every < 1:
        raise ValueError("epochs, batch_size, learning_rate and log_every must be positive")
    if not len(train_rows) or not len(val_rows):
        raise ValueError("Training and validation splits must both be nonempty")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; load its best/ checkpoint or choose a new run directory")
    pairs = PairDataset(train_rows)
    if not len(pairs):
        raise ValueError("No training passages")
    set_seed(config.seed)
    reranker = CrossEncoderReranker(model, max_length=config.max_length, device=device)

    def collate(items):
        features = reranker._tokenize([i["question"] for i in items], [i["passage"] for i in items])
        labels = torch.tensor([i["label"] for i in items], device=reranker.device)
        return features, labels

    loader = DataLoader(
        pairs, batch_size=config.batch_size, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        reranker.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(json.dumps({"model": str(model), **asdict(config)}, indent=2))
    history, best = [], (-1.0, -1.0)
    for epoch in range(1, config.epochs + 1):
        reranker.model.train()
        total_loss, count = 0.0, 0
        for step, (features, labels) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = reranker.model(**features).logits.reshape(-1)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            count += len(labels)
            if step == 1 or step % config.log_every == 0 or step == len(loader):
                print(f"epoch {epoch}/{config.epochs} step {step}/{len(loader)} loss={total_loss / count:.4f}")
        metrics = ev.evaluate(reranker, val_rows)
        record = {"epoch": epoch, "train_loss": total_loss / count, "val": metrics}
        history.append(record)
        candidate = (metrics["all_gold@5"], metrics["recall@5"])
        if candidate > best:
            best = candidate
            reranker.save(output_dir / "best")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"validation all_gold@5={candidate[0]:.3f} recall@5={candidate[1]:.3f}")
    (output_dir / "complete.json").write_text(json.dumps({"best_all_gold@5": best[0], "best_recall@5": best[1]}))
    # The last epoch remains in memory; callers explicitly reload best/ for dev.
    return history
