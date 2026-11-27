# Hybrid cross-encoder + GNN reranking

Run [hybrid.ipynb](../notebooks/hybrid.ipynb) after completing the cross-encoder
experiment for the same data. The hybrid trains one model per dataset, just
like the GNN notebook. It scores evidence passages; it does not generate answers.

## What changes from the GNN?

The graph connections, two GAT layers, loss, and checkpoint-selection metric
are the same. Each passage receives one extra node feature: the frozen
cross-encoder's raw relevance logit.

```text
Question + passage text -> frozen fine-tuned cross-encoder -> CE logit
                                                               |
Passage embedding + question cosine ----------------------------+
                                                               |
                                             770 node features (BGE-base)
                                                               |
                                              4-head GAT: 4 x 64
                                                               |
                                                   ReLU + dropout
                                                               |
                                                GAT: one output
                                                               |
                                                 Passage score
```

The standalone GNN has 769 inputs per node; the hybrid has 770. The CE logit
is not converted with sigmoid, thresholded, or replaced with a supporting label.
Edges still come exclusively from top-3 passage embedding neighbors, made
bidirectional and deduplicated. Both GAT layers add self-loops.

This is stage-wise training: the fine-tuned CE is frozen, the BGE embeddings
are frozen, and a fresh GAT learns to combine their information. Existing GNN
weights are not loaded into the hybrid. The architecture has no additional
residual scoring path and is not guaranteed to beat its component models.

## Files

| File | Responsibility |
| --- | --- |
| [hybrid_scores.py](../src/gnn_rerank/hybrid_scores.py) | Export/cache verified frozen CE predictions |
| [hybrid.py](../src/gnn_rerank/hybrid.py) | Append the CE feature, batch graphs, score passages, save/load checkpoints |
| [train_hybrid.py](../src/gnn_rerank/train_hybrid.py) | Train the hybrid and choose the best validation checkpoint |
| [graph.py](../src/gnn_rerank/graph.py) | Shared graph construction and verified BGE caching |
| [gnn.py](../src/gnn_rerank/gnn.py) | Shared PassageGAT architecture |
| [train_gnn.py](../src/gnn_rerank/train_gnn.py) | Shared TrainConfig, seeding, and epoch-training routine |
| [eval.py](../src/gnn_rerank/eval.py) | Shared retrieval metrics and output score format |

The baseline, cross-encoder, and standalone GNN implementations are unchanged.
The hybrid has its own files and output directories.

## Before running

1. Commit/push the new files and set the notebook's BRANCH to that branch.
2. Use the same Colab GPU, GitHub secret, and Drive shortcut as the other notebooks.
3. Complete cross-encoder training for every selected dataset and the same LIMIT.
4. Use identical prepared data and preprocessing settings across methods.

A full run requires full-data CE checkpoints. A smoke checkpoint is not silently
used for full data. Section 5 checks completed-run markers, dataset/LIMIT, and
hashes of the actual prepared train/val/dev rows against the CE experiment.

For the initial small run:

```python
DATASET_NAMES = ('hotpotqa', '2wiki', 'musique')
LIMIT = 64
RUN_ID = 'smoke-v1'
CROSS_ENCODER_RUN_ID = 'smoke-v1'
GNN_RUN_ID = 'smoke-v1'
```

For the full-data comparison:

```python
LIMIT = None
RUN_ID = 'full-v1'
CROSS_ENCODER_RUN_ID = 'full-v1'
GNN_RUN_ID = 'full-v1'
```

RUN_ID names the new hybrid run. The other IDs identify existing CE and GNN
runs. They can differ if those experiments used different names. Standalone GNN
checkpoints are not needed for hybrid training; their saved scores are only
needed for the comparison table.

## Notebook stages

| Section | Action | Expected output |
| --- | --- | --- |
| 1–4 | GPU, clone, package install, Drive | Setup information |
| 5 | Load prepared data and locate completed CE checkpoints | Split sizes and frozen CE paths |
| 6 | Load BGE vectors and export frozen CE train/val predictions | Cache shapes and scored question counts |
| 7 | Inspect one hybrid graph | Same edges as GNN, one extra feature per passage |
| 8 | Train hybrid GATs on train and select checkpoints on val | Loss, validation metrics, best checkpoint paths |
| 9 | Export CE dev features and score dev with the selected hybrid | Hybrid and source CE metrics |
| 10 | Read existing predictions and compare methods | Combined table, saved metrics and score files |

## Why export CE scores again?

The original CE notebook saved only dev predictions. The hybrid also needs
train and validation predictions from the same selected CE checkpoint. Old
score files contain positional scores but no checkpoint or input fingerprints.
The hybrid therefore uses a separate verified cache:

```text
results/hybrid_features/<dataset_tag>/<CROSS_ENCODER_RUN_ID>/
    train.jsonl
    train.meta.json
    val.jsonl
    val.meta.json
    dev.jsonl
    dev.meta.json
```

Each JSONL record contains question_id, method, scores, and an input_hash. Each
metadata file stores the source checkpoint fingerprint and score-file checksum.
Inputs are checked using question text and ordered passage IDs/titles/text;
labels are excluded. Missing, changed, or incomplete caches are rebuilt using
frozen CE inference. Matching caches load without running the CE again.

Exporting full training predictions can take substantial time. It does not
fine-tune the CE and does not overwrite its old score files or weights. Checkpoint
fingerprints include weights, tokenizer, configuration, and truncation settings.

## Training and evaluation

Training uses all passages, unweighted binary cross-entropy, AdamW, and the same
default five epochs/batch size 32/learning rate 0.001 as the GNN. Only GAT weights
are optimizer parameters. Multiple question graphs are batched without edges
between questions.

Validation Complete Support Recall@5 (`all_gold@5`) selects the best epoch;
Recall@5 breaks ties. Dev is scored afterward. Completed runs are reused only
if settings, train/val data, and actual CE feature values match. Interrupted
training requires a new RUN_ID; optimizer resume is not implemented.

The selected CE's own training predictions are used for hybrid training. This
is not out-of-fold stacking, and the models are not jointly trained. Those are
possible later experiments, not part of this initial hybrid.

The comparison's fine-tuned CE row is evaluated from the exact verified dev
features fed to the hybrid. Other rows are read from existing score files.
Missing comparison files are skipped. Legacy comparison files still require
identical passage order because they lack passage fingerprints.

## Saved hybrid outputs

`<dataset_tag>` is, for example, `musique_limit64` or `musique` for full data.

```text
checkpoints/hybrid/<dataset_tag>/<RUN_ID>/
    experiment.json
    history.json
    best.pt
    complete.json

results/hybrid/<dataset_tag>/<RUN_ID>/
    hybrid_dev.jsonl
    comparison.md
    metrics.json
```

Checkpoint settings include the embedding encoder and frozen CE source identity.
Scoring a reloaded model still requires BGE vectors and CE predictions for that
split; loading rejects a different CE source.

Final predictions use the common score format:

```json
{"question_id": "q1", "method": "hybrid", "scores": [2.1, 1.7, -0.8]}
```

These are raw logits, in prepared passage order, with higher meaning more
relevant. Rerunning section 9 replaces this hybrid run's predictions without
retraining. Baseline, CE, and standalone GNN results are not overwritten.

For tests, run `uv run pytest`. Synthetic checks cover feature/edge alignment,
label isolation, source/cache validation, training updates, validation selection,
checkpoint round trips, and preservation of upstream model weights.
