"""Turn the small local run into data.js for the explainer (+ RQ2/RQ3 previews)."""
import json, sys, re, random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from datasets import load_from_disk, load_dataset, disable_caching
disable_caching()
from gnn_rerank import data, graph, gnn, hybrid, hybrid_scores, eval as ev

RUN, OUTJS = Path(sys.argv[1]), Path(sys.argv[2])
PICK = sys.argv[3].split(",") if len(sys.argv) > 3 else None
ds = load_from_disk(str(RUN / "data")); dev = list(ds["dev"])
emb = graph.GraphEmbeddings(**torch.load(RUN / "emb" / "gnn" / "dev.pt", weights_only=True))
scores = json.loads((RUN / "dev_scores.json").read_text())
metrics = json.loads((RUN / "metrics.json").read_text())
src = hybrid_scores.checkpoint_hash(RUN / "ce" / "best")
feat = hybrid_scores.load_scores(ds["dev"], RUN / "hyb_feat" / "dev.jsonl", source=src)
G = gnn.GNNReranker.load(RUN / "gnn" / "best.pt", emb, device="cpu")
H = hybrid.HybridReranker.load(RUN / "hybrid" / "best.pt", emb, feat, device="cpu")
G.model.eval(); H.model.eval()
METHODS = ["cosine", "cross_encoder", "gnn", "hybrid"]
K = 3

def und(edge_index):
    return sorted({(min(a, b), max(a, b)) for a, b in edge_index.T.tolist()})
def to_index(pairs):
    if not pairs: return torch.empty((2, 0), dtype=torch.long)
    e = torch.tensor(pairs).T
    return torch.cat([e, e.flip(0)], 1)
def components(nodes, pairs):
    parent = {n: n for n in nodes}
    def find(a):
        while parent[a] != a: a = parent[a]
        return a
    for a, b in pairs:
        if a in parent and b in parent: parent[find(a)] = find(b)
    return len({find(n) for n in nodes})
def conn_class(gold, pairs):
    c = components(gold, pairs)
    return "connected" if c == 1 else "disconnected" if c == len(gold) else "partial"
def ag5(row, s): return ev.question_metrics(row, s)["all_gold@5"]

# per-question graphs
info = []
for i, row in enumerate(dev):
    g = graph.build_graph(row, emb, graph.GraphConfig(K))
    hg = hybrid.build_hybrid_graph(row, emb, feat, graph.GraphConfig(K))
    gold = [j for j, p in enumerate(row["passages"]) if p["is_supporting"]]
    pairs = und(g.edge_index)
    info.append(dict(g=g, hg=hg, gold=gold, pairs=pairs, cls=conn_class(gold, pairs),
                     ok={m: ag5(row, scores[m][i]) for m in scores}))

# RQ2 preview
conn = []
for cls in ("connected", "partial", "disconnected"):
    idx = [i for i, x in enumerate(info) if x["cls"] == cls]
    conn.append(dict(cls=cls, n=len(idx), **{m: (ev.evaluate_scores([dev[i] for i in idx], [scores[m][i] for i in idx])["all_gold@5"] if idx else None) for m in METHODS}))
print("RQ2", json.dumps(conn, indent=1))

# RQ3 preview
def corrupt(pairs, n, mode, rate, rng):
    m = round(rate * len(pairs))
    if mode == "delete":
        drop = set(rng.sample(range(len(pairs)), m)); return [p for j, p in enumerate(pairs) if j not in drop]
    have = set(pairs); non = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in have]
    return pairs + rng.sample(non, min(m, len(non)))
noise = {}
RATES = [0, .1, .2, .3, .4, .5]
with torch.inference_mode():
    for mode in ("delete", "insert"):
        noise[mode] = {"gnn": [], "hybrid": []}
        for rate in RATES:
            tot = {"gnn": 0.0, "hybrid": 0.0}; SEEDS = 5
            for seed in range(SEEDS):
                rng = random.Random(f"{mode}:{rate}:{seed}")
                for i, row in enumerate(dev):
                    e = to_index(corrupt(info[i]["pairs"], len(row["passages"]), mode, rate, rng))
                    tot["gnn"] += ag5(row, G.model(info[i]["g"].x, e).tolist())
                    tot["hybrid"] += ag5(row, H.model(info[i]["hg"].x, e).tolist())
            for m in tot: noise[mode][m].append(tot[m] / (SEEDS * len(dev)))
print("RQ3", json.dumps(noise))

# manual GAT forward (reference for the in-browser version)
def params(model):
    sd = {k: v.numpy() for k, v in model.state_dict().items()}
    return sd
def manual(model, x, pairs, n):
    sd = params(model); x = x.numpy()
    nb = [[i] for i in range(n)]
    for a, b in pairs: nb[a].append(b); nb[b].append(a)
    W1 = sd["first.lin.weight"]; h = x @ W1.T
    heads = sd["first.att_src"].shape[1]; C = sd["first.att_src"].shape[2]
    hh = h.reshape(n, heads, C)
    s = (hh * sd["first.att_src"][0]).sum(-1); d = (hh * sd["first.att_dst"][0]).sum(-1)
    out = np.zeros_like(hh)
    for i in range(n):
        for hd in range(heads):
            e = np.array([s[j, hd] + d[i, hd] for j in nb[i]]); e = np.where(e > 0, e, 0.2 * e)
            a = np.exp(e - e.max()); a /= a.sum()
            out[i, hd] = sum(a[t] * hh[j, hd] for t, j in enumerate(nb[i]))
    hid = np.maximum(out.reshape(n, heads * C) + sd["first.bias"], 0)
    z = (hid @ sd["second.lin.weight"].T)[:, 0]
    s2 = z * sd["second.att_src"].ravel()[0]; d2 = z * sd["second.att_dst"].ravel()[0]
    res = []
    for i in range(n):
        e = np.array([s2[j] + d2[i] for j in nb[i]]); e = np.where(e > 0, e, 0.2 * e)
        a = np.exp(e - e.max()); a /= a.sum()
        res.append(float(sum(a[t] * z[j] for t, j in enumerate(nb[i])) + sd["second.bias"][0]))
    return res, h
print("state keys", {k: tuple(v.shape) for k, v in G.model.state_dict().items()})
ref, _ = manual(G.model, info[0]["g"].x, info[0]["pairs"], len(dev[0]["passages"]))
print("manual vs torch max diff", max(abs(a - b) for a, b in zip(ref, scores["gnn"][0])))

# candidates for worked examples
raw = {r["id"]: r for r in load_dataset(data.MUSIQUE_REPO, data_files={"dev": "musique_ans_v1.0_dev.jsonl"}, split="dev")}
def show(tag, cond, limit=8):
    print(f"\n== {tag}")
    c = [i for i, x in enumerate(info) if cond(x, dev[i])]
    print(len(c), "candidates")
    for i in c[:limit]:
        o = info[i]["ok"]
        print(f"  {dev[i]['question_id']} hops={dev[i]['hops']} cls={info[i]['cls']} " + " ".join(f"{m[:3]}={int(o[m])}" for m in METHODS) + f" | {dev[i]['question'][:110]} -> {dev[i]['answer']}")
if PICK is None:
    show("lead: cosine fails, gnn+hybrid succeed, connected, 2hop", lambda x, r: not x["ok"]["cosine"] and x["ok"]["gnn"] and x["ok"]["hybrid"] and x["cls"] == "connected" and r["hops"] == 2, 14)
    show("CE ok, GNN fails", lambda x, r: x["ok"]["cross_encoder"] and not x["ok"]["gnn"], 8)
    show("hybrid ok, CE and GNN fail", lambda x, r: x["ok"]["hybrid"] and not x["ok"]["cross_encoder"] and not x["ok"]["gnn"], 8)
    show("disconnected", lambda x, r: x["cls"] == "disconnected", 10)
    show("partial", lambda x, r: x["cls"] == "partial", 10)
    show("3-4 hop connected, cos fail, hybrid ok", lambda x, r: x["cls"] == "connected" and r["hops"] >= 3 and not x["ok"]["cosine"] and x["ok"]["hybrid"], 8)
    show("all succeed 2hop", lambda x, r: all(x["ok"][m] for m in METHODS) and r["hops"] == 2, 4)
    sys.exit()

def layout(sim):
    n = len(sim); D = np.sqrt(np.maximum(2 - 2 * sim, 0)); J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J; w, v = np.linalg.eigh(B)
    xy = v[:, -2:] * np.sqrt(np.maximum(w[-2:], 1e-9)); xy = xy[:, ::-1]
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9) * np.array([360, 270])
    for _ in range(400):
        moved = False
        for a in range(n):
            for b in range(a + 1, n):
                d = xy[b] - xy[a]; dist = np.linalg.norm(d)
                if dist < 46:
                    push = (46 - dist) / 2 * (d / dist if dist > 1e-6 else np.array([1.0, 0.0]))
                    xy[a] -= push; xy[b] += push; moved = True
        if not moved: break
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0)); return np.round(xy, 3).tolist()

def rnd(a, k=3): return np.round(np.asarray(a, dtype=float), k).tolist()
examples = []
for qid in PICK:
    i = next(j for j, r in enumerate(dev) if r["question_id"] == qid); row = dev[i]
    q, P = emb.get(qid); p = F.normalize(P.float(), dim=1); sim = (p @ p.T).numpy()
    n = len(row["passages"])
    assert und(graph.knn_edges(p, K)) == info[i]["pairs"]
    pos = {pp["doc_id"]: j for j, pp in enumerate(row["passages"])}
    steps, answers = [], []
    for st in raw[qid]["question_decomposition"]:
        text = re.sub(r"#(\d+)", lambda m: answers[int(m.group(1)) - 1] if int(m.group(1)) <= len(answers) else m.group(0), st["question"])
        answers.append(st["answer"])
        steps.append({"q": text, "a": st["answer"], "p": pos.get(f"{qid}_{st['paragraph_support_idx']}")})
    _, h_g = manual(G.model, info[i]["g"].x, info[i]["pairs"], n)
    _, h_h = manual(H.model, info[i]["hg"].x, info[i]["pairs"], n)
    examples.append(dict(
        id=qid, question=row["question"], answer=row["answer"], hops=row["hops"], cls=info[i]["cls"],
        passages=[dict(title=pp["title"], text=pp["text"][:900], gold=bool(pp["is_supporting"])) for pp in row["passages"]],
        steps=steps, sim=rnd(sim, 4), xy=layout(sim), k3=info[i]["pairs"],
        scores={m: rnd(scores[m][i], 6) for m in ("cosine", "ce_pretrained", "cross_encoder", "gnn", "gnn_k0", "hybrid")},
        h={"gnn": rnd(h_g, 5), "hybrid": rnd(h_h, 5)},
    ))
def model_params(model):
    sd = params(model)
    return dict(heads=int(sd["first.att_src"].shape[1]), C=int(sd["first.att_src"].shape[2]),
                as1=rnd(sd["first.att_src"][0], 6), ad1=rnd(sd["first.att_dst"][0], 6), b1=rnd(sd["first.bias"], 6),
                w2=rnd(sd["second.lin.weight"][0], 6), as2=float(sd["second.att_src"].ravel()[0]), ad2=float(sd["second.att_dst"].ravel()[0]), b2=float(sd["second.bias"][0]))
hist = lambda p: json.loads((RUN / p / "history.json").read_text())
out = dict(
    run=dict(dataset="MuSiQue-Ans", n_train=len(ds["train"]), n_val=len(ds["val"]), n_dev=len(dev), k=K,
             ce_epochs=len(hist("ce")), gnn_epochs=len(hist("gnn"))),
    metrics={m: {k: round(v, 4) for k, v in metrics[m].items()} for m in metrics},
    conn=conn, noise=dict(rates=RATES, **noise, flat={m: metrics[m]["all_gold@5"] for m in ("cosine", "cross_encoder")}),
    models={"gnn": model_params(G.model), "hybrid": model_params(H.model)}, examples=examples,
)
OUTJS.write_text("window.DATA = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("wrote", OUTJS, OUTJS.stat().st_size // 1024, "KB")
