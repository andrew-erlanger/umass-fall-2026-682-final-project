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
        "question_type": str,   # hotpotqa: bridge / comparison; 2wiki: its type field; musique: 2hop / 3hop / 4hop
        "hops": int,            # reasoning hops: musique from its id, otherwise the number of gold passages
    }

``doc_id`` is ``"<question_id>_<i>"``, unique across the whole dataset.
The official test sets are unlabeled, so ``dev`` serves as our test set and
``split_train_val`` carves a validation set out of ``train``.

Pass ``limit=N`` to any loader to stream only the first N questions per split
instead of downloading the full dataset (for smoke tests and debugging).

None of these datasets are gated: no Hugging Face token is needed.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from datasets import Dataset, DatasetDict, load_dataset

HOTPOTQA_REPO = "hotpotqa/hotpot_qa"
MUSIQUE_REPO = "bdsaglam/musique"
TWOWIKI_REPO = "framolfese/2WikiMultihopQA"


def _passage(qid: str, i: int, title: str, text: str, supporting: bool) -> dict:
    return {"doc_id": f"{qid}_{i}", "title": title, "text": text, "is_supporting": supporting}


def _row(
    qid: str, question: str, answer: str, aliases: list[str], passages: list[dict], dataset: str,
    question_type: str = "unknown", hops: int | None = None,
) -> dict:
    gold = [p["doc_id"] for p in passages if p["is_supporting"]]
    return {
        "question_id": qid,
        "question": question,
        "answer": answer,
        "answer_aliases": aliases,
        "passages": passages,
        "gold_support_ids": gold,
        "dataset": dataset,
        "question_type": question_type or "unknown",
        "hops": hops if hops is not None else len(gold),
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
    return _row(ex["id"], ex["question"], ex["answer"], [], passages, dataset, question_type=ex.get("type") or "unknown")


def _musique_hops(ex: dict) -> int:
    """MuSiQue ids start with the hop count ("2hop__…", "3hop1__…", "4hop3__…");
    fall back to the length of the question decomposition."""
    m = re.match(r"(\d)hop", ex["id"])
    return int(m.group(1)) if m else len(ex.get("question_decomposition") or [])


def _normalize_musique(ex: dict) -> dict:
    passages = [
        _passage(ex["id"], i, p["title"], p["paragraph_text"], p["is_supporting"])
        for i, p in enumerate(ex["paragraphs"])
    ]
    hops = _musique_hops(ex)
    return _row(
        ex["id"], ex["question"], ex["answer"], list(ex["answer_aliases"]), passages, "musique",
        question_type=f"{hops}hop", hops=hops,
    )


def _finish(raw: DatasetDict, fn: Callable[[dict], dict]) -> DatasetDict:
    return DatasetDict(
        {split: ds.map(fn, remove_columns=ds.column_names, desc=f"normalize {split}") for split, ds in raw.items()}
    )


def _load_raw(repo: str, splits: dict[str, str], limit: int | None, **kwargs) -> DatasetDict:
    """Load ``{our_split: their_split}``; with ``limit``, stream just the first rows."""
    if limit is None:
        raw = load_dataset(repo, **kwargs)
        return DatasetDict({ours: raw[theirs] for ours, theirs in splits.items()})
    return DatasetDict(
        {
            ours: Dataset.from_list(list(load_dataset(repo, split=theirs, streaming=True, **kwargs).take(limit)))
            for ours, theirs in splits.items()
        }
    )


def load_hotpotqa(limit: int | None = None) -> DatasetDict:
    """HotpotQA, distractor setting: 2 gold + 8 distractor paragraphs per question."""
    raw = _load_raw(HOTPOTQA_REPO, {"train": "train", "dev": "validation"}, limit, name="distractor")
    return _finish(raw, lambda ex: _normalize_wiki_style(ex, "hotpotqa"))


def load_2wiki(limit: int | None = None) -> DatasetDict:
    """2WikiMultiHopQA: 10 paragraphs per question, 2-4 of them supporting."""
    raw = _load_raw(TWOWIKI_REPO, {"train": "train", "dev": "validation"}, limit)
    return _finish(raw, lambda ex: _normalize_wiki_style(ex, "2wiki"))


def load_musique(limit: int | None = None) -> DatasetDict:
    """MuSiQue-Ans (answerable subset): 20 paragraphs per question, 2-4 hops."""
    files = {"train": "musique_ans_v1.0_train.jsonl", "dev": "musique_ans_v1.0_dev.jsonl"}
    raw = _load_raw(MUSIQUE_REPO, {"train": "train", "dev": "dev"}, limit, data_files=files)
    return _finish(raw, _normalize_musique)


LOADERS: dict[str, Callable[..., DatasetDict]] = {
    "hotpotqa": load_hotpotqa,
    "2wiki": load_2wiki,
    "musique": load_musique,
}


def load(name: str, limit: int | None = None) -> DatasetDict:
    """Load a dataset by name: one of ``hotpotqa``, ``2wiki``, ``musique``."""
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(LOADERS)}")
    return LOADERS[name](limit=limit)


def split_train_val(ds: DatasetDict, val_fraction: float = 0.1, seed: int = 682) -> DatasetDict:
    """Split ``train`` into ``train``/``val``; ``dev`` is kept as the test set."""
    parts = ds["train"].train_test_split(test_size=val_fraction, seed=seed)
    return DatasetDict({"train": parts["train"], "val": parts["test"], "dev": ds["dev"]})
