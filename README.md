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
5. In Drive, add a shortcut to the shared `cs682-final` folder in **My Drive** (*Shared with me* →
   right-click → *Organize → Add shortcut*).
6. **Runtime → Run all.** The notebook checks the GPU, clones the repo, installs it, mounts Drive and
   loads the datasets.

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

```
src/gnn_rerank/
  data.py          dataset loaders, all normalized to one schema (see module docstring)
notebooks/
  starter.ipynb    Colab entry point
```

## Data

| Dataset | Train / dev questions | Passages per question | Gold per question |
| --- | --- | --- | --- |
| HotpotQA (distractor) | 90,447 / 7,405 | 10 | 2 |
| 2WikiMultiHopQA | 167,454 / 12,576 | 10 | 2–4 |
| MuSiQue-Ans | 19,938 / 2,417 | 20 | 2–4 |

The official test sets are unlabeled, so **dev is our test set**. `data.split_train_val()` holds out
10% of train for validation (seed 682).

```python
from gnn_rerank import data
ds = data.split_train_val(data.load("musique"))   # train / val / dev
```
