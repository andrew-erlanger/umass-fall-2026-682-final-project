"""Shared evaluation. Every number in the report comes from here.

Metrics, per question, averaged over questions (gold = the supporting passages):

- ``recall@k``: fraction of gold passages ranked in the top k
- ``all_gold@k``: 1 if *every* gold passage is in the top k (the multi-hop metric:
  a reader can only answer if it sees all hops)
- ``r_precision``: recall at k = (number of gold passages for that question)
- ``all_gold@G``: 1 if all gold passages fill the top G slots, G = that question's
  gold count. Prefer this to ``all_gold@2`` on 2Wiki/MuSiQue, where questions with
  3-4 gold passages can never score on ``all_gold@2``
- ``mrr``: 1 / rank of the first gold passage
- ``ndcg@10``: binary-relevance nDCG over the top 10

Two ways to use it::

    eval.evaluate(reranker, ds["dev"])          # run a Reranker and score it
    eval.evaluate_scores(rows, scores)          # score saved scores (no model needed)

Ties are broken by passage position. That would reward a constant scorer
whenever gold passages tend to come first, so use ``RandomReranker`` as the
lower bound, not a constant.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_KS = (1, 2, 5, 10)


def ranking(scores: Sequence[float]) -> list[int]:
    """Passage indices sorted best-first (stable on ties)."""
    return sorted(range(len(scores)), key=lambda j: -scores[j])


def question_metrics(row: dict, scores: Sequence[float], ks: Sequence[int] = DEFAULT_KS) -> dict[str, float]:
    passages = row["passages"]
    if len(scores) != len(passages):
        raise ValueError(f"{row['question_id']}: {len(scores)} scores for {len(passages)} passages")
    gold = [p["is_supporting"] for p in passages]
    n_gold = sum(gold)
    if n_gold == 0:
        raise ValueError(f"{row['question_id']}: no gold passages")

    hits = [gold[j] for j in ranking(scores)]  # hits[r] = is the passage at rank r+1 gold?
    out = {}
    for k in ks:
        found = sum(hits[:k])
        out[f"recall@{k}"] = found / n_gold
        out[f"all_gold@{k}"] = float(found == n_gold)
    found_g = sum(hits[:n_gold])
    out["r_precision"] = found_g / n_gold
    out["all_gold@G"] = float(found_g == n_gold)
    out["mrr"] = 1.0 / (hits.index(True) + 1)
    dcg = sum(1.0 / math.log2(r + 2) for r, h in enumerate(hits[:10]) if h)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(n_gold, 10)))
    out["ndcg@10"] = dcg / idcg
    return out


def evaluate_scores(rows: Iterable[dict], scores: Iterable[Sequence[float]], ks: Sequence[int] = DEFAULT_KS) -> dict:
    """Mean of each metric over questions, plus ``n`` (number of questions)."""
    totals: dict[str, float] = {}
    n = 0
    for row, s in zip(rows, scores, strict=True):
        for name, v in question_metrics(row, s, ks).items():
            totals[name] = totals.get(name, 0.0) + v
        n += 1
    if n == 0:
        raise ValueError("no questions to evaluate")
    return {**{name: v / n for name, v in totals.items()}, "n": n}


def _batches(ds, batch_size: int):
    batch = []
    for row in ds:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def score_all(reranker, ds, batch_size: int = 64) -> list[list[float]]:
    """Run ``reranker`` over every row of ``ds``; one score list per row."""
    out: list[list[float]] = []
    for batch in _batches(ds, batch_size):
        scores = reranker.score(batch)
        if len(scores) != len(batch):
            raise ValueError(f"{reranker.name}: {len(scores)} score lists for a batch of {len(batch)}")
        out.extend([float(x) for x in s] for s in scores)
    return out


def evaluate(reranker, ds, ks: Sequence[int] = DEFAULT_KS, batch_size: int = 64, scores_path: str | Path | None = None) -> dict:
    """Score ``ds`` with ``reranker`` and compute the metrics. With ``scores_path``,
    also save the raw scores (for the hybrid and for error analysis)."""
    scores = score_all(reranker, ds, batch_size)
    if scores_path is not None:
        save_scores(scores_path, ds, scores, reranker.name)
    return evaluate_scores(ds, scores, ks)


def evaluate_by(rows, scores, key: str = "question_type", ks: Sequence[int] = DEFAULT_KS) -> dict[str, dict]:
    """Metrics per group, e.g. per question type or per hop count: ``{group: metrics}``."""
    groups: dict[str, tuple[list, list]] = {}
    for row, s in zip(rows, scores, strict=True):
        g = groups.setdefault(str(row[key]), ([], []))
        g[0].append(row)
        g[1].append(s)
    return {g: evaluate_scores(r, s, ks) for g, (r, s) in sorted(groups.items())}


# --- saved scores: how tracks hand results to each other (e.g. to the hybrid) ---


def save_scores(path: str | Path, rows: Iterable[dict], scores: Iterable[Sequence[float]], method: str) -> None:
    """One JSON line per question: ``{"question_id", "method", "scores"}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for row, s in zip(rows, scores, strict=True):
            f.write(json.dumps({"question_id": row["question_id"], "method": method, "scores": list(s)}) + "\n")
    tmp.replace(path)


def load_scores(path: str | Path) -> dict[str, list[float]]:
    """``{question_id: scores}`` from a file written by ``save_scores``."""
    with open(path) as f:
        return {(r := json.loads(line))["question_id"]: r["scores"] for line in f}


def aligned(ds, saved: dict[str, list[float]]) -> list[list[float]]:
    """Saved scores in ``ds`` row order; raises if any question is missing."""
    missing = [qid for qid in ds["question_id"] if qid not in saved]
    if missing:
        raise KeyError(f"{len(missing)} questions have no saved scores, e.g. {missing[:3]}")
    return [saved[qid] for qid in ds["question_id"]]


# --- the results table ---


def results_table(results: dict[tuple[str, str], dict], metrics: Sequence[str] = ("r_precision", "all_gold@G", "all_gold@5", "mrr", "ndcg@10")) -> str:
    """Markdown table from ``{(method, dataset): metrics}``, one row per pair."""
    lines = ["| Method | Dataset | " + " | ".join(metrics) + " |", "|" + " --- |" * (len(metrics) + 2)]
    for (method, dataset), m in results.items():
        lines.append(f"| {method} | {dataset} | " + " | ".join(f"{m[k]:.3f}" for k in metrics) + " |")
    return "\n".join(lines)
