"""The shared reranker interface, plus the reference baselines.

Every method in the project (cosine, cross-encoder, GNN, hybrid) implements
``Reranker``: given a batch of normalized rows (see ``data.py``), return one list
of scores per row, aligned with ``row["passages"]``. Higher means more relevant.
``eval.evaluate`` only relies on this, so any reranker plugs into the same metrics.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .embed import Embeddings


@runtime_checkable
class Reranker(Protocol):
    name: str

    def score(self, rows: Sequence[dict]) -> list[list[float]]:
        """One score per passage, for each row, in ``row["passages"]`` order."""
        ...


class RandomReranker:
    """Lower bound: random scores, seeded per question so runs are reproducible."""

    name = "random"

    def __init__(self, seed: int = 682):
        self.seed = seed

    def score(self, rows: Sequence[dict]) -> list[list[float]]:
        out = []
        for row in rows:
            rng = random.Random(f"{self.seed}:{row['question_id']}")
            out.append([rng.random() for _ in row["passages"]])
        return out


class CosineReranker:
    """Baseline: cosine similarity between the question and passage embeddings.

    Reads precomputed embeddings (``embed.cached``), so scoring needs no GPU. The
    embeddings are unit-normalized, so a dot product is the cosine similarity.
    """

    name = "cosine"

    def __init__(self, emb: Embeddings):
        self.emb = emb

    def score(self, rows: Sequence[dict]) -> list[list[float]]:
        out = []
        for row in rows:
            q, P = self.emb.get(row["question_id"])
            if P.shape[0] != len(row["passages"]):
                raise ValueError(f"{row['question_id']}: {P.shape[0]} cached passages, row has {len(row['passages'])}")
            cached_ids = self.emb.doc_ids_of(row["question_id"])
            if cached_ids is not None and cached_ids != [p["doc_id"] for p in row["passages"]]:
                raise ValueError(f"{row['question_id']}: cached passages are in a different order; rebuild the embedding cache")
            out.append((P.float() @ q.float()).tolist())
        return out
