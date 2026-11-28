# Known issues

Found while reviewing and running PR #2 (cross-encoder, GNN, hybrid) on 28 Nov 2026. None blocked the merge.
Evidence comes from a small local run: 1,350 MuSiQue training questions, 300 dev questions.

## Affects results

| # | Issue | Where | Evidence | Fix |
|---|---|---|---|---|
| 1 | **Hybrid trains on leaked scores.** The cross-encoder scores the same questions it was trained on, so its scores are better in training than on dev. | `hybrid_scores.cached_scores`, `hybrid.ipynb` | Small run: hybrid all_gold@5 51.0%, cross-encoder alone 61.7% | Score train out-of-fold: train a second cross-encoder on the other half |
| 2 | **Tied GNN scores.** Passages with the same set of neighbours get exactly the same score. `GATConv` attention depends only on the sending node, so a node's own features do not set it apart. | `gnn.PassageGAT` | 6 of 6 same-neighbourhood pairs tied in 7 dev questions. Small run: GNN 31.3%, cosine 36.3%, GNN with no edges 34.3% | Skip connection, or `GATv2Conv`, or pass edge weights |
| 3 | **Edges carry no weight.** Passage-to-passage cosine picks the edges and is then discarded. | `graph.build_graph` | By reading the code | Pass cosine as `edge_attr` |

## Affects running

| # | Issue | Where | Fix |
|---|---|---|---|
| 4 | **Wrong branch in a notebook.** `BRANCH = 'justin/dev'`, which has no `cross_encoder.py`; the notebook fails at import. | `notebooks/cross_encoder.ipynb` | Point all three notebooks at `master` |
| 5 | **No resume for cross-encoder training.** An interrupted run cannot continue, and a non-empty run folder is refused. Full 2Wiki is about 1.5M pairs per epoch. | `cross_encoder.train` | Save optimizer state per N steps, or run outside Colab |
| 6 | **Crash window.** `experiment.json` is written after training; if the session dies in between, the next run fails on the missing file. | `notebooks/cross_encoder.ipynb` | Write the manifest before training |
| 7 | **Notebooks lack cell IDs.** They show as modified on every checkout under nbstripout. | `cross_encoder.ipynb`, `gnn.ipynb`, `hybrid.ipynb` | Normalize once with nbformat |
| 8 | **Duplicate embedding cache.** The GNN writes its own verified copy under `gnn/`. The shared cache now has the same check (this branch), so the copy is redundant. | `graph.cached_embeddings` | Read the shared cache directly |
| 9 | **Stale local dataset cache.** `datasets.map` can reuse a cached result from older loader code and return rows without `question_type`. Local runs only; fresh Colab VMs are unaffected. | `data._finish` | `datasets.disable_caching()` or clear the HF cache |

## Checked and fine

| Check | Result |
|---|---|
| Tests on PR #2 | 33/33 pass |
| Tests with this branch rebased on PR #2 | 39/39 pass |
| Gold labels in scoring or edge selection | Not used |
| Dev split in model selection | Not used; all three models select on validation |
| End-to-end run on a laptop (M4 Pro, MPS) | Works; 13.5 minutes for the small run |
