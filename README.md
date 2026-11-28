# CS 682 Final Project: GNN vs. cross-encoder reranking for multi-hop RAG

We compare passage rerankers (FAISS cosine baseline, fine-tuned cross-encoder, GNN, and a hybrid)
on multi-hop QA: HotpotQA (distractor), 2WikiMultiHopQA and MuSiQue-Ans.

Code lives in this repo. Experiments run in Colab on an A100. Large artifacts (embeddings,
checkpoints) live in the shared Google Drive folder.

## Running in Colab

1. Open the starter notebook:
   [`notebooks/starter.ipynb`](https://colab.research.google.com/github/andrew-erlanger/umass-fall-2026-682-final-project/blob/master/notebooks/starter.ipynb).
   Because the repo is private, Colab asks you to authorize GitHub the first time. Tick
   **"Include private repos"**.
2. **File → Save a copy in Drive.** Everyone works in their own copy. Don't edit the shared starter
   notebook in place.
3. **Runtime → Change runtime type → A100 GPU.** It isn't the default.
4. In the 🔑 **Secrets** panel, add:
   - `GITHUB_TOKEN`: your own
     [fine-grained token](https://github.com/settings/personal-access-tokens/new) with
     *Contents: Read-only* access to this repo. Turn on notebook access for it.
   - `HF_TOKEN` (optional): our datasets are public, but a token gives faster downloads.
5. Open the shared Drive folder,
   [`umass-fall-2026-682-final-project`](https://drive.google.com/drive/folders/1GjBXAOSqHJspuhrfvnRj1VXr4uEwXP12),
   and add it to **My Drive**: click the folder name → *Organize → Add shortcut* → *My Drive*. Keep
   the shortcut's name unchanged, because the notebook looks for `MyDrive/umass-fall-2026-682-final-project`.
6. **Runtime → Run all.** The notebook checks the GPU, clones the repo, installs it, mounts Drive,
   loads a small batch of each dataset (`LIMIT = 64`), and runs a GPU smoke test. The smoke test embeds
   passages, builds passage graphs, trains a GAT for a few steps, and writes `smoke_test.txt` to Drive.
   Set `LIMIT = None` to load the full datasets.

The Colab VM is wiped when the session ends. Save anything worth keeping to the Drive folder.

## Local setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:andrew-erlanger/umass-fall-2026-682-final-project.git
cd umass-fall-2026-682-final-project
uv sync
uv run nbstripout --install   # once per clone: strips notebook outputs on commit
```

## Making changes

- All code changes go through a **branch and pull request**. Don't push directly to `master`.
- Put reusable code in `src/gnn_rerank/`. Keep notebooks thin: they should call into the package.
- Notebook outputs are stripped on commit (nbstripout), so commit notebooks freely without merge
  conflicts over cell output.
- To try a branch in Colab, set `BRANCH` in the notebook's clone cell and rerun it. After a `git pull`,
  autoreload picks up code changes without reinstalling.
- Add dependencies with `uv add <pkg>`, and commit `pyproject.toml` and `uv.lock`. Don't pin `torch`,
  so Colab keeps its preinstalled build.

## Layout

Diagrams of the data model, modules, caches and API: [`docs/architecture.md`](docs/architecture.md).

```
src/gnn_rerank/
  data.py          dataset loaders, all normalized to one schema (see module docstring)
  preprocess.py    cleaning, shuffling, train/val split and Drive cache: use `prepare()`, not `data.load()`
  embed.py         shared embedding cache on Drive (one encoder for every method)
  rerank.py        the Reranker interface every method implements, plus random/cosine baselines
  eval.py          the shared metrics; every reported number comes from here
  cross_encoder.py text-pair scoring, fine-tuning, and checkpoint saving/loading
  graph.py         verified passage features and bidirectional cosine-neighbor graphs
  gnn.py           GAT architecture, scoring, and checkpoint saving/loading
  train_gnn.py     training loop and validation checkpoint selection
  hybrid_scores.py verified frozen cross-encoder predictions for hybrid features
  hybrid.py        GNN graphs with an extra cross-encoder score per passage
  train_hybrid.py  hybrid training using the shared GAT/epoch routine
notebooks/
  starter.ipynb    Colab entry point
  baselines.ipynb  builds the embedding cache and the baseline results table
  cross_encoder.ipynb trains a cross-encoder and compares it with saved baselines
  gnn.ipynb        inspects graphs, trains GNNs, and compares saved predictions
  hybrid.ipynb     exports CE features, trains hybrids, and compares all methods
docs/
  architecture.md  system map, data model, API
  diagrams/        the SVGs it shows (hand-written; edit as text)
  gnn.md           GNN layers, training, notebook stages, saved files
  hybrid.md        hybrid steps, cache checks, saved files
  known-issues.md  open problems in the cross-encoder, GNN and hybrid code
tests/             `uv run pytest`
  test_data.py        loader normalization
  test_preprocess.py  cleaning and prepare() cache
  test_embed.py       embedding cache staleness
  test_eval.py        metrics and saved scores
  test_graph.py       topology, batching, label isolation, and embedding alignment
  test_gnn.py         real GAT training, score ordering, and checkpoint checks
```

## Cross-encoder experiment

Use `notebooks/cross_encoder.ipynb` in Colab with the same token and Drive setup.
Set `BRANCH` to the pushed branch containing this notebook and `cross_encoder.py`;
local edits are not available to Colab until committed and pushed. Run sections in
order. All three datasets run separately. Start with `LIMIT = 64` to check the pipeline, then use
`LIMIT = None` and a new `RUN_ID` for full training. No embedding cache is required.

The starting checkpoint is
[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
Training uses all question/passage pairs with supporting labels and binary
cross-entropy. Validation `all_gold@5` selects the best epoch (`recall@5` breaks
ties); dev is used only for the final comparison. Defaults are one epoch, pair
batch size 16, learning rate 2e-5, and 512 tokens per question/passage pair.
These are starting settings, not tuned experiment results.

Checkpoints and training history go to
`checkpoints/cross_encoder/<dataset>[_limitN]/<RUN_ID>/` on Drive. Load `best/`
to score with the selected model. Completed runs with matching settings/data are
reused; interrupted training needs a new run directory (optimizer resume is not
implemented). Dev scores and comparison tables go to
`results/cross_encoder/<dataset>[_limitN]/<RUN_ID>/`. Existing random/cosine score
files are read for comparison, not overwritten; they must use the same prepared
data and passage order. The notebook does not implement answer generation/EM/F1.

## GNN experiment

Run `notebooks/gnn.ipynb` in Colab with the same data/token/Drive setup as
baselines. Set `BRANCH` to your pushed branch. The notebook follows a linear
flow: load data, load embeddings, inspect a graph, train one GAT per dataset,
score dev, and compare existing predictions. Start with `LIMIT = 64` and use
`LIMIT = None` plus a new `RUN_ID` for full runs.

Model code lives in `gnn.py`, graph construction in `graph.py`, and training in
`train_gnn.py`. GNN-specific verified embedding copies leave original baseline
caches untouched. The self-only comparison is a separate optional run.
See [docs/gnn.md](docs/gnn.md) for layers, training, notebook stages, and saved files.

## Hybrid experiment

Run `notebooks/hybrid.ipynb` after the matching cross-encoder run has completed.
Set LIMIT and CROSS_ENCODER_RUN_ID to that experiment, and choose a RUN_ID for
the new hybrid. All three datasets run separately. For a full run use
`LIMIT = None`, `RUN_ID = "full-v1"`, and the matching CE/GNN run IDs.

The hybrid appends the frozen CE's raw score to each GNN node (770 inputs with
BGE-base). It reuses graph construction, PassageGAT, and the GNN epoch-training
routine. Only the new GAT trains. Train/val CE predictions are generated and
cached separately because the CE notebook originally exports only dev scores.
Original baseline/CE/GNN implementations and outputs are unchanged.
See [docs/hybrid.md](docs/hybrid.md) for steps, cache checks, and saved files.

## Adding a method

Implement the `Reranker` interface in `rerank.py`: a `name` and a `score(rows)` that returns one score per
passage for each row, higher meaning more relevant. Then:

```python
from gnn_rerank import eval as ev
metrics = ev.evaluate(my_reranker, ds["dev"], scores_path=DRIVE_DIR / "results/scores/<name>_<dataset>_dev.jsonl")
```

Tune on `val`, never on `dev`. Saving the scores lets the hybrid and the error analysis reuse them.
Report `all_gold@k` alongside recall: a multi-hop question is only answerable if *all* its gold passages make the cut.

## Data

| Dataset | Train / dev questions | Passages per question | Gold per question |
| --- | --- | --- | --- |
| HotpotQA (distractor) | 90,447 / 7,405 | 10 | 2 |
| 2WikiMultiHopQA | 167,454 / 12,576 | 10 | 2–4 |
| MuSiQue-Ans | 19,938 / 2,417 | 20 | 2–4 |

The official test sets are unlabeled, so **dev is our test set**. `data.split_train_val()` holds out
10% of train for validation (seed 682).

```python
from gnn_rerank import preprocess
ds, report = preprocess.prepare("musique", cache_dir=DRIVE_DIR / "prepared")   # train / val / dev, cleaned
```

`prepare` normalizes whitespace, removes empty and duplicate passages, drops questions with fewer than 2 gold
passages, and shuffles passage order with a fixed per-question seed. Everyone should load data this way, so all methods
see identical questions. `report.json` in the cache folder lists what was dropped and summary statistics.
