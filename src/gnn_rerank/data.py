"""Dataset loaders.

Each loader returns a ``DatasetDict`` with ``train`` and ``dev`` splits, every
row normalized to one shared schema::

    {
        "question_id": str,
        "question": str,
        "answer": str,
        "answer_aliases": list[str],
        "passages": [{"doc_id": str, "title": str, "text": str, "is_supporting": bool}],
        "gold_support_ids": list[str],
        "dataset": str,
    }

``doc_id`` is ``"<question_id>_<i>"``, unique across the whole dataset.
The official test sets are unlabeled, so ``dev`` serves as our test set and
``split_train_val`` carves a validation set out of ``train``.

None of these datasets are gated: no Hugging Face token is needed.
"""

from __future__ import annotations

from collections.abc import Callable

from datasets import DatasetDict, load_dataset

HOTPOTQA_REPO = "hotpotqa/hotpot_qa"
MUSIQUE_REPO = "bdsaglam/musique"
TWOWIKI_REPO = "framolfese/2WikiMultihopQA"


def _passage(qid: str, i: int, title: str, text: str, supporting: bool) -> dict:
    return {"doc_id": f"{qid}_{i}", "title": title, "text": text, "is_supporting": supporting}


def _row(qid: str, question: str, answer: str, aliases: list[str], passages: list[dict], dataset: str) -> dict:
    return {
        "question_id": qid,
        "question": question,
        "answer": answer,
        "answer_aliases": aliases,
        "passages": passages,
        "gold_support_ids": [p["doc_id"] for p in passages if p["is_supporting"]],
        "dataset": dataset,
    }


def _normalize_wiki_style(ex: dict, dataset: str) -> dict:
    """HotpotQA and 2Wiki share a format: 10 titled paragraphs given as sentence
    lists, and supporting facts given as (title, sentence index) pairs."""
    gold_titles = set(ex["supporting_facts"]["title"])
    ctx = ex["context"]
    passages = [
        _passage(ex["id"], i, title, " ".join(s.strip() for s in sents), title in gold_titles)
        for i, (title, sents) in enumerate(zip(ctx["title"], ctx["sentences"]))
    ]
    return _row(ex["id"], ex["question"], ex["answer"], [], passages, dataset)


def _normalize_musique(ex: dict) -> dict:
    passages = [
        _passage(ex["id"], i, p["title"], p["paragraph_text"], p["is_supporting"])
        for i, p in enumerate(ex["paragraphs"])
    ]
    return _row(ex["id"], ex["question"], ex["answer"], list(ex["answer_aliases"]), passages, "musique")


def _finish(raw: DatasetDict, fn: Callable[[dict], dict]) -> DatasetDict:
    return DatasetDict(
        {split: ds.map(fn, remove_columns=ds.column_names, desc=f"normalize {split}") for split, ds in raw.items()}
    )


def load_hotpotqa() -> DatasetDict:
    """HotpotQA, distractor setting: 2 gold + 8 distractor paragraphs per question."""
    raw = load_dataset(HOTPOTQA_REPO, "distractor")
    raw = DatasetDict({"train": raw["train"], "dev": raw["validation"]})
    return _finish(raw, lambda ex: _normalize_wiki_style(ex, "hotpotqa"))


def load_2wiki() -> DatasetDict:
    """2WikiMultiHopQA: 10 paragraphs per question, 2-4 of them supporting."""
    raw = load_dataset(TWOWIKI_REPO)
    raw = DatasetDict({"train": raw["train"], "dev": raw["validation"]})
    return _finish(raw, lambda ex: _normalize_wiki_style(ex, "2wiki"))


def load_musique() -> DatasetDict:
    """MuSiQue-Ans (answerable subset): 20 paragraphs per question, 2-4 hops."""
    raw = load_dataset(
        MUSIQUE_REPO,
        data_files={"train": "musique_ans_v1.0_train.jsonl", "dev": "musique_ans_v1.0_dev.jsonl"},
    )
    return _finish(raw, _normalize_musique)


LOADERS: dict[str, Callable[[], DatasetDict]] = {
    "hotpotqa": load_hotpotqa,
    "2wiki": load_2wiki,
    "musique": load_musique,
}


def load(name: str) -> DatasetDict:
    """Load a dataset by name: one of ``hotpotqa``, ``2wiki``, ``musique``."""
    try:
        return LOADERS[name]()
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(LOADERS)}") from None


def split_train_val(ds: DatasetDict, val_fraction: float = 0.1, seed: int = 682) -> DatasetDict:
    """Split ``train`` into ``train``/``val``; ``dev`` is kept as the test set."""
    parts = ds["train"].train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict({"train": parts["train"], "val": parts["test"], "dev": ds["dev"]})
