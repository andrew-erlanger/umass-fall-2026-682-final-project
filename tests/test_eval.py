"""Checks for the shared metrics. Run: ``uv run pytest`` (or ``python tests/test_eval.py``)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_rerank import eval as ev
from gnn_rerank.rerank import RandomReranker


def row(gold, qid="q"):
    return {
        "question_id": qid,
        "passages": [{"doc_id": f"{qid}_{i}", "is_supporting": g} for i, g in enumerate(gold)],
        "gold_support_ids": [f"{qid}_{i}" for i, g in enumerate(gold) if g],
    }


def test_perfect_ranking():
    m = ev.question_metrics(row([0, 1, 0, 1]), [0.1, 0.9, 0.2, 0.8])
    assert m["recall@1"] == 0.5 and m["recall@2"] == 1.0
    assert m["all_gold@1"] == 0.0 and m["all_gold@2"] == 1.0
    assert m["mrr"] == 1.0 and math.isclose(m["ndcg@10"], 1.0)


def test_worst_ranking():
    m = ev.question_metrics(row([1, 0, 0, 0]), [0.0, 1.0, 0.9, 0.8])
    assert m["recall@2"] == 0.0 and m["mrr"] == 0.25
    assert math.isclose(m["ndcg@10"], 1 / math.log2(5))


def test_ties_are_stable():
    assert ev.ranking([0.5, 0.5, 0.9]) == [2, 0, 1]


def test_evaluate_scores_averages():
    rows = [row([1, 0], "a"), row([0, 1], "b")]
    m = ev.evaluate_scores(rows, [[1, 0], [1, 0]])
    assert m["n"] == 2 and m["recall@1"] == 0.5 and m["mrr"] == 0.75


def test_random_reranker_reproducible_and_evaluable():
    rows = [row([1, 0, 1, 0, 0], f"q{i}") for i in range(50)]
    r = RandomReranker()
    assert r.score(rows) == r.score(rows)
    m = ev.evaluate(r, rows, batch_size=7)
    assert m["n"] == 50 and 0 < m["recall@2"] < 1


def test_saved_scores_roundtrip(tmp_path=None):
    import tempfile
    rows = [row([1, 0], "a"), row([0, 1], "b")]
    path = Path(tmp_path or tempfile.mkdtemp()) / "s.jsonl"
    ev.save_scores(path, rows, [[0.3, 0.7], [0.1, 0.2]], "x")
    saved = ev.load_scores(path)
    assert ev.aligned({"question_id": ["b", "a"]}, saved) == [[0.1, 0.2], [0.3, 0.7]]


def test_length_mismatch_raises():
    try:
        ev.question_metrics(row([1, 0]), [1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
