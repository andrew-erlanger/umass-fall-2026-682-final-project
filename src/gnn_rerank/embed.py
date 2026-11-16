"""Compute passage and question embeddings once and cache them on Drive.

Every method that needs embeddings (cosine baseline, GNN node features) loads
the same cache, so the comparison is fair and nobody re-embeds every session::

    emb = embed.cached(ds["dev"], EMB_DIR / "musique" / "dev.pt")
    q, P = emb.get(question_id)   # (d,), (n_passages, d); unit-normalized, fp16

Passage ``j`` of a question is ``P[j]``, in ``row["passages"]`` order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

# Some encoders expect an instruction in front of queries (not passages).
QUERY_PREFIX = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def passage_text(p: dict) -> str:
    return f"{p['title']}: {p['text']}"


@dataclass
class Embeddings:
    model: str
    question_ids: list[str]
    questions: torch.Tensor  # (n_questions, d)
    passages: torch.Tensor  # (n_passages_total, d)
    offsets: torch.Tensor  # (n_questions + 1,); question i owns passages[offsets[i]:offsets[i+1]]

    def __post_init__(self):
        self.index = {qid: i for i, qid in enumerate(self.question_ids)}

    def get(self, question_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``(question_embedding, passage_embeddings)`` for one question."""
        if question_id not in self.index:
            raise KeyError(f"{question_id!r} not in this cache; was it built from the same split?")
        i = self.index[question_id]
        return self.questions[i], self.passages[self.offsets[i] : self.offsets[i + 1]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "model": self.model,
                "question_ids": self.question_ids,
                "questions": self.questions,
                "passages": self.passages,
                "offsets": self.offsets,
            },
            tmp,
        )
        tmp.replace(path)  # atomic: a crashed Colab session never leaves a half-written cache

    @classmethod
    def load(cls, path: str | Path) -> Embeddings:
        return cls(**torch.load(path, map_location="cpu"))


def embed_split(ds, model: str = DEFAULT_MODEL, batch_size: int = 256, device: str | None = None) -> Embeddings:
    """Embed every question and passage in one split (a ``datasets.Dataset``)."""
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = SentenceTransformer(model, device=device)
    if device == "cuda":
        enc.half()

    passages = ds["passages"]
    texts = [passage_text(p) for ps in passages for p in ps]
    sizes = torch.tensor([0] + [len(ps) for ps in passages])
    prefix = QUERY_PREFIX.get(model, "")

    kw = dict(batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    Q = enc.encode([prefix + q for q in ds["question"]], **kw)
    P = enc.encode(texts, **kw)
    return Embeddings(
        model=model,
        question_ids=list(ds["question_id"]),
        questions=Q.half().cpu(),
        passages=P.half().cpu(),
        offsets=torch.cumsum(sizes, 0),
    )


def cached(ds, path: str | Path, model: str = DEFAULT_MODEL, **kwargs) -> Embeddings:
    """Load ``path`` if it holds embeddings of exactly this split with this model;
    otherwise compute them and save to ``path``."""
    path = Path(path)
    if path.exists():
        emb = Embeddings.load(path)
        if emb.model == model and emb.question_ids == list(ds["question_id"]):
            return emb
        print(f"{path}: stale cache (different model or questions), recomputing")
    emb = embed_split(ds, model=model, **kwargs)
    emb.save(path)
    return emb
