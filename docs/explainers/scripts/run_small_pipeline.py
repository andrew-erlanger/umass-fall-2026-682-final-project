"""Small local run of the team's pipeline on a MuSiQue sample (for the explainer)."""
import json, sys, time, random
from pathlib import Path
import torch
from datasets import DatasetDict, load_from_disk, disable_caching
disable_caching()  # a stale local map cache predates question_type/hops
from gnn_rerank import data, preprocess, embed, graph, gnn, train_gnn, hybrid, hybrid_scores, train_hybrid, rerank
from gnn_rerank import cross_encoder as ce, eval as ev

OUT = Path(sys.argv[1]); N_TRAIN, N_DEV = 1500, 300
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ENC = "BAAI/bge-base-en-v1.5"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# A. data
dpath = OUT / "data"
if not dpath.exists():
    raw = data.load("musique")
    rng = random.Random(682)
    tr = raw["train"].select(sorted(rng.sample(range(len(raw["train"])), N_TRAIN)))
    dv = raw["dev"].select(sorted(rng.sample(range(len(raw["dev"])), N_DEV)))
    ds = data.split_train_val(DatasetDict({"train": preprocess.clean(tr)[0], "dev": preprocess.clean(dv)[0]}))
    ds.save_to_disk(str(dpath))
ds = load_from_disk(str(dpath))
log("data", {s: len(d) for s, d in ds.items()})

# B. embeddings
emb = {s: graph.cached_embeddings(ds[s], OUT / "emb" / f"{s}.pt", model=ENC, device=DEV, batch_size=64) for s in ds}
log("embeddings done")

scores = {}
scores["random"] = ev.score_all(rerank.RandomReranker(), ds["dev"])
scores["cosine"] = ev.score_all(rerank.CosineReranker(emb["dev"]), ds["dev"])

# D. cross-encoder: pretrained, then fine-tuned
pre = OUT / "ce_pre_dev.json"
if not pre.exists():
    r = ce.CrossEncoderReranker(device=DEV, batch_size=64)
    pre.write_text(json.dumps(ev.score_all(r, ds["dev"]))); del r
scores["ce_pretrained"] = json.loads(pre.read_text())
log("pretrained CE scored")
if not (OUT / "ce" / "complete.json").exists():
    ce.train(ds["train"], ds["val"], OUT / "ce", config=ce.TrainConfig(epochs=2, log_every=200), device=DEV)
log("CE trained", (OUT / "ce" / "history.json").read_text()[:0])
src = hybrid_scores.checkpoint_hash(OUT / "ce" / "best")
feat = {s: hybrid_scores.cached_scores(ds[s], OUT / "ce" / "best", OUT / "hyb_feat" / f"{s}.jsonl", source=src, batch_size=64, device=DEV) for s in ds}
scores["cross_encoder"] = [feat["dev"].get(row) for row in ds["dev"]]
log("CE features exported")

# E. GNN (k=3) and the no-edges ablation (k=0)
cfg = train_gnn.TrainConfig(epochs=30, log_every=1000)
for tag, k in (("gnn", 3), ("gnn_k0", 0)):
    gc = graph.GraphConfig(neighbors=k)
    train_gnn.train(ds["train"], ds["val"], emb["train"], emb["val"], OUT / tag, graph_config=gc, config=cfg, device="cpu")
    scores[tag] = ev.score_all(gnn.GNNReranker.load(OUT / tag / "best.pt", emb["dev"], device="cpu"), ds["dev"])
log("GNN trained")

# F. hybrid
train_hybrid.train(ds["train"], ds["val"], emb["train"], emb["val"], feat["train"], feat["val"], OUT / "hybrid", config=cfg, device="cpu")
scores["hybrid"] = ev.score_all(hybrid.HybridReranker.load(OUT / "hybrid" / "best.pt", emb["dev"], feat["dev"], device="cpu"), ds["dev"])
log("hybrid trained")

(OUT / "dev_scores.json").write_text(json.dumps(scores))
metrics = {m: ev.evaluate_scores(ds["dev"], s) for m, s in scores.items()}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(ev.results_table({(m, "musique"): v for m, v in metrics.items()}, metrics=("recall@5", "recall@10", "all_gold@5", "all_gold@G", "mrr")))
log("DONE")
