"""Checks for passage cleaning. Run: ``uv run pytest``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import Dataset, DatasetDict

from gnn_rerank import preprocess
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


def test_hops_follows_gold_count_after_merge():
    ps = [P(0, "A", "x", True), P(1, "A", "x", True), P(2, "B", "y", True), P(3, "C", "z", False)]
    wiki = clean_row(row(ps) | {"dataset": "2wiki", "hops": 3})
    assert wiki["hops"] == 2  # the two copies of A merged
    assert clean_row(row(ps) | {"dataset": "musique", "hops": 3})["hops"] == 3  # from the id, untouched


def test_drop_reasons():
    assert clean_row(row([P(0, "A", "x", True), P(1, "B", "y", False)]))["_drop"] == "fewer_than_2_gold"
    assert clean_row(row([P(0, "A", "x", True), P(1, "B", "y", True)]))["_drop"] == "all_passages_gold"
    assert clean_row(row([P(0, "A", "", True), P(1, "B", "y", True), P(2, "C", "z", True), P(3, "D", "w", False)]))["_drop"] == "empty_gold_passage"


def test_shuffle_is_deterministic_and_optional():
    ps = [P(0, "A", "a", True), P(1, "B", "b", True)] + [P(i, f"T{i}", f"t{i}", False) for i in range(2, 20)]
    a, b = clean_row(row(ps)), clean_row(row(ps))
    assert a["passages"] == b["passages"] and [p["doc_id"] for p in a["passages"]] != [p["doc_id"] for p in ps]
    assert [p["doc_id"] for p in clean_row(row(ps), shuffle=False)["passages"]] == [p["doc_id"] for p in ps]


def test_prepare_cache_respects_settings(tmp_path, monkeypatch):
    def fake_load(name, limit=None):
        rows = [row([P(i * 10 + j, f"T{j}", f"t{i} {j}", j < 2) for j in range(6)]) | {"question_id": f"q{i}", "question_type": "x"}
                for i in range(20)]
        return DatasetDict({"train": Dataset.from_list(rows[:15]), "dev": Dataset.from_list(rows[15:])})
    loads = []
    monkeypatch.setattr(preprocess.data, "load", lambda *a, **k: loads.append(1) or fake_load(*a, **k))
    preprocess.prepare("fake", cache_dir=tmp_path)
    preprocess.prepare("fake", cache_dir=tmp_path)
    assert len(loads) == 1  # cache hit
    _, report = preprocess.prepare("fake", cache_dir=tmp_path, shuffle=False)
    assert len(loads) == 2 and report["shuffle"] is False  # different settings: rebuilt, not silently reused


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
