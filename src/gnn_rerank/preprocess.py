"""Preprocessing between the raw loaders (``data.py``) and every method.

``prepare`` is the one call every track should use::

    from gnn_rerank import preprocess
    ds, report = preprocess.prepare("musique", cache_dir=DRIVE_DIR / "prepared")
    # ds: DatasetDict with train / val / dev, same schema as data.py

It loads the dataset, cleans each split (``clean``), splits train into train/val
and caches the result on Drive, so later sessions skip all of it. ``report``
records what was changed or dropped and summary statistics for the report's
Data section; it is saved next to the cache as ``report.json``.

What ``clean`` does to each question:

1. Normalizes whitespace in titles, passage text and questions.
2. Drops passages with empty text (unless supporting; then the question is dropped).
3. Merges exact duplicate passages (same title and text), keeping the first copy
   and marking it supporting if any copy was.
4. Drops questions with fewer than ``min_gold`` supporting passages (multi-hop
   needs at least 2) or where every passage is supporting (nothing to rerank).
5. Shuffles passage order with a per-question seed, so no method (or the
   evaluator's tie-breaking) can benefit from where the datasets put gold
   passages. ``doc_id`` keeps the original index, so passages stay traceable.

Before step 5, the report records where gold passages sat, which tells you
whether the shuffle mattered.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_from_disk

from . import data

# Bump this whenever data.py or this module changes what a prepared row contains:
# prepare() rebuilds any Drive cache written by a different version.
VERSION = 2

_WS = re.compile(r"\s+")
LONG_WORDS = 300  # roughly 400+ tokens: at risk of truncation by 512-token encoders


def _norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def clean_row(row: dict, min_gold: int = 2, shuffle: bool = True, seed: int = 682) -> dict:
    """Clean one question. Returns the row plus ``_drop`` (a reason, or "") and
    ``_stats`` (what changed), which ``clean`` aggregates and then removes."""
    stats = {"empty": 0, "duplicate": 0, "gold_positions": [], "n_in": len(row["passages"])}
    drop = ""
    seen: dict[tuple[str, str], dict] = {}
    passages = []
    for pos, p in enumerate(row["passages"]):
        if p["is_supporting"]:
            stats["gold_positions"].append(pos)
        title, text = _norm(p["title"]), _norm(p["text"])
        if not text:
            stats["empty"] += 1
            if p["is_supporting"]:
                drop = "empty_gold_passage"
            continue
        key = (title, text)
        if key in seen:
            stats["duplicate"] += 1
            seen[key]["is_supporting"] = seen[key]["is_supporting"] or p["is_supporting"]
            continue
        q = {"doc_id": p["doc_id"], "title": title, "text": text, "is_supporting": p["is_supporting"]}
        seen[key] = q
        passages.append(q)

    n_gold = sum(p["is_supporting"] for p in passages)
    if not drop and n_gold < min_gold:
        drop = f"fewer_than_{min_gold}_gold"
    elif not drop and n_gold == len(passages):
        drop = "all_passages_gold"

    if shuffle:
        random.Random(f"{seed}:{row['question_id']}").shuffle(passages)

    return {
        **row,
        "question": _norm(row["question"]),
        "passages": passages,
        "gold_support_ids": [p["doc_id"] for p in passages if p["is_supporting"]],
        "_drop": drop,
        "_stats": json.dumps(stats),
    }


def _hist(c: Counter) -> dict[str, int]:
    """Counter -> {str(key): count}, sorted by key (str keys survive the JSON round trip)."""
    return {str(k): v for k, v in sorted(c.items())}


def _summarize(cleaned) -> dict:
    drops = Counter(r for r in cleaned["_drop"] if r)
    stats = [json.loads(s) for s in cleaned["_stats"]]
    kept = [ps for ps, d in zip(cleaned["passages"], cleaned["_drop"]) if not d]
    words = sorted(len(p["text"].split()) for ps in kept for p in ps)
    gold_pos = Counter(pos for s in stats for pos in s["gold_positions"])
    return {
        "questions_in": len(cleaned),
        "questions_kept": len(kept),
        "dropped": dict(drops),
        "passages_removed": {"empty": sum(s["empty"] for s in stats), "duplicate": sum(s["duplicate"] for s in stats)},
        "passages_per_question": _hist(Counter(len(ps) for ps in kept)),
        "gold_per_question": _hist(Counter(sum(p["is_supporting"] for p in ps) for ps in kept)),
        "question_types": _hist(Counter(t for t, d in zip(cleaned["question_type"], cleaned["_drop"]) if not d)),
        # where gold sat in the original order: if this is far from uniform, the shuffle mattered
        "gold_position_before_shuffle": _hist(gold_pos),
        "passage_words": {
            "mean": round(sum(words) / len(words), 1) if words else 0,
            "p50": words[len(words) // 2] if words else 0,
            "p95": words[int(len(words) * 0.95)] if words else 0,
            "max": words[-1] if words else 0,
            f"frac_over_{LONG_WORDS}": round(sum(w > LONG_WORDS for w in words) / len(words), 4) if words else 0,
        },
    }


def clean(ds, min_gold: int = 2, shuffle: bool = True, seed: int = 682):
    """Clean one split (a ``datasets.Dataset``). Returns ``(dataset, report)``."""
    cleaned = ds.map(
        clean_row, fn_kwargs=dict(min_gold=min_gold, shuffle=shuffle, seed=seed), desc="clean"
    )
    report = _summarize(cleaned)
    kept = cleaned.filter(lambda d: not d, input_columns="_drop", desc="drop")
    return kept.remove_columns(["_drop", "_stats"]), report


def prepare(
    name: str,
    limit: int | None = None,
    cache_dir: str | Path | None = None,
    min_gold: int = 2,
    shuffle: bool = True,
    seed: int = 682,
) -> tuple[DatasetDict, dict]:
    """Load ``name``, clean every split, split train into train/val, and cache.

    With ``cache_dir``, the result lives in ``cache_dir/<name>[_limitN]/`` and is
    loaded from there next time. Delete that folder to rebuild (e.g. after
    changing this module)."""
    out_dir = None
    settings = {"version": VERSION, "dataset": name, "limit": limit, "min_gold": min_gold, "shuffle": shuffle, "seed": seed}
    if cache_dir is not None:
        out_dir = Path(cache_dir) / (name if limit is None else f"{name}_limit{limit}")
        if (out_dir / "report.json").exists():
            report = json.loads((out_dir / "report.json").read_text())
            cached_settings = {k: report.get(k) for k in settings}
            if cached_settings == settings:
                return load_from_disk(str(out_dir / "data")), report
            diff = {k: (cached_settings[k], v) for k, v in settings.items() if cached_settings[k] != v}
            print(f"{out_dir}: cache built with different settings {diff} (cached, requested), rebuilding")

    raw = data.load(name, limit=limit)
    splits, report = {}, dict(settings)
    for split, ds in raw.items():
        splits[split], report[split] = clean(ds, min_gold=min_gold, shuffle=shuffle, seed=seed)
    ds = data.split_train_val(DatasetDict(splits), seed=seed)
    report["sizes"] = {split: len(d) for split, d in ds.items()}

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").unlink(missing_ok=True)  # a crash mid-rebuild must not leave a "complete" marker
        ds.save_to_disk(str(out_dir / "data"))
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))  # written last: marks the cache complete
    return ds, report
