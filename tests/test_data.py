"""Checks for the loaders' normalization (no downloads). Run: ``uv run pytest``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_rerank import data


def wiki_example(**extra):
    return {
        "id": "x1", "question": "q", "answer": "a",
        "context": {"title": ["A", "B", "C"], "sentences": [[" a1", " a2"], ["b1"], ["c1"]]},
        "supporting_facts": {"title": ["A", "C"], "sent_id": [0, 0]},
        **extra,
    }


def test_wiki_style_metadata():
    r = data._normalize_wiki_style(wiki_example(type="bridge"), "hotpotqa")
    assert r["question_type"] == "bridge" and r["hops"] == 2
    assert r["gold_support_ids"] == ["x1_0", "x1_2"] and r["passages"][0]["text"] == "a1 a2"
    assert data._normalize_wiki_style(wiki_example(), "2wiki")["question_type"] == "unknown"


def test_musique_hops():
    ex = {"id": "3hop1__123_456_789", "question": "q", "answer": "a", "answer_aliases": [],
          "paragraphs": [{"title": "T", "paragraph_text": "p", "is_supporting": True}] * 3}
    r = data._normalize_musique(ex)
    assert r["hops"] == 3 and r["question_type"] == "3hop"
    assert data._musique_hops({"id": "weird", "question_decomposition": [1, 2, 3, 4]}) == 4
