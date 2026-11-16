"""Checks for the embedding cache's staleness detection (no model download)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_rerank import embed
from gnn_rerank.rerank import CosineReranker


def split(order):
    """One question whose passages appear in ``order`` (a list of original indices)."""
    ps = [{"doc_id": f"q_{i}", "title": f"T{i}", "text": f"t{i}", "is_supporting": i < 2} for i in order]
    return {"question_id": ["q"], "question": ["?"], "passages": [ps]}


def fake_embed_split(ds, model=embed.DEFAULT_MODEL, **_):
    ps = ds["passages"][0]
    return embed.Embeddings(
        model=model,
        question_ids=list(ds["question_id"]),
        questions=torch.ones(1, 2),
        passages=torch.tensor([[float(p["doc_id"][2:]), 0.0] for p in ps]),
        offsets=torch.tensor([0, len(ps)]),
        doc_ids=[p["doc_id"] for p in ps],
    )


def test_cache_rebuilds_when_passage_order_changes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(embed, "embed_split", lambda ds, **kw: calls.append(1) or fake_embed_split(ds, **kw))
    path = tmp_path / "dev.pt"
    embed.cached(split([0, 1, 2]), path)
    embed.cached(split([0, 1, 2]), path)
    assert len(calls) == 1  # second call hit the cache
    emb = embed.cached(split([2, 0, 1]), path)
    assert len(calls) == 2 and emb.doc_ids_of("q") == ["q_2", "q_0", "q_1"]


def test_cosine_refuses_misaligned_cache():
    emb = fake_embed_split(split([0, 1, 2]))
    rows = [{"question_id": "q", "passages": split([2, 0, 1])["passages"][0]}]
    with pytest.raises(ValueError, match="different order"):
        CosineReranker(emb).score(rows)
