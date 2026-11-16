"""Checks for passage cleaning. Run: ``uv run pytest``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_rerank.preprocess import clean_row


def P(i, title, text, gold):
    return {"doc_id": f"q_{i}", "title": title, "text": text, "is_supporting": gold}


def row(passages):
    return {"question_id": "q", "question": " a  b\n", "passages": passages,
            "gold_support_ids": [p["doc_id"] for p in passages if p["is_supporting"]]}


def test_cleans_merges_and_keeps_gold():
    out = clean_row(row([P(0, "A", "x  y", True), P(1, "A", "x y", False), P(2, "B", "z", True),
                         P(3, "C", "   ", False), P(4, "D", "w", False)]))
    assert out["_drop"] == "" and out["question"] == "a b"
    assert sorted(p["doc_id"] for p in out["passages"]) == ["q_0", "q_2", "q_4"]  # duplicate and empty removed
    assert sorted(out["gold_support_ids"]) == ["q_0", "q_2"]


def test_duplicate_of_gold_stays_gold():
    out = clean_row(row([P(0, "A", "x", False), P(1, "A", "x", True), P(2, "B", "y", True), P(3, "C", "z", False)]))
    assert sorted(out["gold_support_ids"]) == ["q_0", "q_2"]


def test_drop_reasons():
    assert clean_row(row([P(0, "A", "x", True), P(1, "B", "y", False)]))["_drop"] == "fewer_than_2_gold"
    assert clean_row(row([P(0, "A", "x", True), P(1, "B", "y", True)]))["_drop"] == "all_passages_gold"
    assert clean_row(row([P(0, "A", "", True), P(1, "B", "y", True), P(2, "C", "z", True), P(3, "D", "w", False)]))["_drop"] == "empty_gold_passage"


def test_shuffle_is_deterministic_and_optional():
    ps = [P(0, "A", "a", True), P(1, "B", "b", True)] + [P(i, f"T{i}", f"t{i}", False) for i in range(2, 20)]
    a, b = clean_row(row(ps)), clean_row(row(ps))
    assert a["passages"] == b["passages"] and [p["doc_id"] for p in a["passages"]] != [p["doc_id"] for p in ps]
    assert [p["doc_id"] for p in clean_row(row(ps), shuffle=False)["passages"]] == [p["doc_id"] for p in ps]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
