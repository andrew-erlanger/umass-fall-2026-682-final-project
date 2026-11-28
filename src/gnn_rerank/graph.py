"""Build one passage graph per question, without using labels to choose edges.

Graphs stay on CPU until batched for training/inference. Similarity connections
are separate from the model so later experiments can alter graph connectivity.
"""

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from . import embed


def row_hash(row):
    """Fingerprint model inputs and passage order, never supporting labels."""
    inputs = [row["question"], [(p["doc_id"], p["title"], p["text"]) for p in row["passages"]]]
    return hashlib.sha256(json.dumps(inputs, ensure_ascii=False).encode()).hexdigest()


@dataclass
class GraphEmbeddings(embed.Embeddings):
    """The baseline vectors plus input fingerprints needed for graph alignment."""

    row_hashes: list[str] | None = None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        # Keep references to tensors; asdict() would deep-copy a full split.
        torch.save({field.name: getattr(self, field.name) for field in fields(self)}, temporary)
        temporary.replace(path)


def cached_embeddings(rows, baseline_path, *, model=embed.DEFAULT_MODEL, **kwargs):
    """Load verified vectors, keeping GNN-specific caching out of embed.py.

    Old baseline caches cannot prove their passage order. Leave those files
    alone and build a verified copy in a gnn/ subfolder once. Existing verified
    shared caches can be reused directly. Nothing writes to baseline_path.
    """
    baseline_path = Path(baseline_path)
    graph_path = baseline_path.parent / "gnn" / baseline_path.name
    hashes = [row_hash(row) for row in rows]
    ids = list(rows["question_id"])
    for path in (graph_path, baseline_path):
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["model"] == model and saved["question_ids"] == ids and saved.get("row_hashes") == hashes:
                return GraphEmbeddings(**saved)
            del saved  # Release a stale split before encoding its replacement.
    print(f"Building verified GNN embeddings: {graph_path}")
    vectors = embed.embed_split(rows, model=model, **kwargs)
    payload = {field.name: getattr(vectors, field.name) for field in fields(vectors)}
    payload["row_hashes"] = hashes
    verified = GraphEmbeddings(**payload)
    verified.save(graph_path)
    return verified


@dataclass(frozen=True)
class GraphConfig:
    neighbors: int = 3

    def __post_init__(self):
        if not isinstance(self.neighbors, int) or self.neighbors < 0:
            raise ValueError("neighbors must be a nonnegative integer")


def knn_edges(passages: torch.Tensor, neighbors: int = 3) -> torch.Tensor:
    """Union of top-k cosine neighbors in both directions, without self-loops.

    The stored graph is bidirectional and has no duplicate edges. Each GAT
    layer will add its own self-loops. k=0 provides the no-message-passing
    ablation: every node can then only attend to itself inside the GAT.
    """
    GraphConfig(neighbors)  # Validate even when the graph has only one node.
    n = passages.shape[0]
    k = min(neighbors, max(0, n - 1))
    if k == 0:
        return torch.empty((2, 0), dtype=torch.long, device=passages.device)
    vectors = F.normalize(passages.detach().float(), dim=-1)
    similarities = vectors @ vectors.T
    similarities.fill_diagonal_(-torch.inf)
    # Stable ties prefer earlier passage positions. Sorting on CPU makes the
    # edge construction reproducible even when the training device is CUDA.
    nearest = similarities.argsort(dim=1, descending=True, stable=True)[:, :k]
    source = torch.arange(n, device=passages.device).repeat_interleave(k)
    edges = torch.stack([source, nearest.flatten()])
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    return torch.unique(edges, dim=1)  # Remove duplicate source/target columns.


def build_graph(row: dict, emb: GraphEmbeddings, config: GraphConfig | None = None, *, with_labels=False) -> Data:
    """Node j stays aligned with row['passages'][j], including during scoring.

    Features are [normalized passage vector, cosine(question, passage)]. The
    supporting labels are added as y ONLY when explicitly requested for fitting.
    """
    config = config or GraphConfig()
    qid = row["question_id"]
    q, passages = emb.get(qid)
    if emb.row_hashes is None:
        raise ValueError("Embedding cache lacks content/order hashes; use graph.cached_embeddings()")
    if emb.row_hashes[emb.index[qid]] != row_hash(row):
        raise ValueError(f"{qid}: embedding content/order differs from the prepared row")
    if not len(row["passages"]):
        raise ValueError(f"{qid}: cannot build a graph without passages")
    if passages.ndim != 2 or q.ndim != 1 or passages.shape != (len(row["passages"]), q.numel()):
        raise ValueError(f"{qid}: embedding shape does not match passages/question")
    # Work in float32: cached fp16 vectors can have small normalization errors.
    q = F.normalize(q.detach().float().cpu(), dim=0)
    p = F.normalize(passages.detach().float().cpu(), dim=1)
    if not torch.isfinite(q).all() or not torch.isfinite(p).all():
        raise ValueError(f"{qid}: non-finite embeddings")
    features = torch.cat([p, (p @ q).unsqueeze(1)], dim=1)
    graph = Data(x=features, edge_index=knn_edges(p, config.neighbors))
    if with_labels:
        graph.y = torch.tensor([p["is_supporting"] for p in row["passages"]], dtype=torch.float32)
    return graph


class GraphDataset:
    """Construct small graphs on demand instead of caching Python graph objects."""

    def __init__(self, rows, emb, config=None, *, with_labels=False):
        self.rows, self.emb = rows, emb
        self.config = config or GraphConfig()
        self.with_labels = with_labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return build_graph(self.rows[index], self.emb, self.config, with_labels=self.with_labels)


def dataset_hash(rows) -> str:
    """Run identity includes labels too, unlike the embedding input hash."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()
