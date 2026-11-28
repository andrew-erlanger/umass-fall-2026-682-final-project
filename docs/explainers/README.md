# Explainers

Four interactive pages on what the project is and how the method works. Open `index.html` in a browser
(no server or build step needed):

    open docs/explainers/index.html

| File | What it is |
|---|---|
| `index.html` | The pages: text, styles, static diagrams |
| `app.js` | Interactions, including the in-browser GAT forward pass |
| `data.js` | Real questions, graphs, scores and trained weights from a small local run |
| `scripts/` | The two scripts that produced `data.js` |

The numbers come from a small run (1,350 MuSiQue training questions, 300 dev), not the full-data experiments.
To regenerate `data.js`, run from a checkout that has the cross-encoder, GNN and hybrid modules
(currently the `bao/cross-encoder` branch):

    uv run python docs/explainers/scripts/run_small_pipeline.py /tmp/rerank-run
    uv run python docs/explainers/scripts/export_data.py /tmp/rerank-run docs/explainers/data.js <question ids, comma-separated>

Run `export_data.py` without the last argument to list candidate questions.
