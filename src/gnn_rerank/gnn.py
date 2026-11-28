"""GAT architecture, passage scoring, and checkpoint saving/loading.

Only GAT parameters are trained. The scoring adapter and shared evaluator use
the same passage order/score format as cosine and cross-encoder rerankers.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv

from .graph import GraphConfig, build_graph

FEATURE_VERSION = 1


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dim: int = 64
    heads: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.input_dim, self.hidden_dim, self.heads) < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Model dimensions must be positive; dropout must be in [0, 1)")


class PassageGAT(nn.Module):
    """[embedding + query cosine] -> multi-head GAT -> ReLU/dropout -> logit."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.first = GATConv(config.input_dim, config.hidden_dim, heads=config.heads, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.second = GATConv(config.hidden_dim * config.heads, 1, heads=1, dropout=config.dropout)

    def forward(self, x, edge_index):
        # GATConv adds self-loops, so isolated nodes still retain their own input.
        hidden = self.first(x, edge_index)  # [N, input_dim] -> [N, heads * hidden_dim]
        hidden = hidden.relu()
        hidden = self.dropout(hidden)
        scores = self.second(hidden, edge_index)  # [N, heads * hidden_dim] -> [N, 1]
        return scores.squeeze(-1)  # One logit per passage: [N]


class GNNReranker:
    name = "gnn"

    def __init__(self, model, emb, graph_config=None, *, device=None, batch_size=64, name="gnn"):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if model.config.input_dim != emb.questions.shape[1] + 1:
            raise ValueError("Checkpoint input dimension does not match the embedding model")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.emb = emb
        self.graph_config = graph_config or GraphConfig()
        self.batch_size, self.name = batch_size, name

    def score(self, rows):
        """Never reads supporting labels; split batched node logits back by graph."""
        self.model.eval()
        out = []
        for start in range(0, len(rows), self.batch_size):
            graphs = [build_graph(row, self.emb, self.graph_config) for row in rows[start:start + self.batch_size]]
            batch = Batch.from_data_list(graphs).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch.x, batch.edge_index).float().cpu()
            # PyG offsets node IDs during batching; there are no cross-question
            # edges. ptr marks where each question's original node order starts.
            pointers = batch.ptr.cpu().tolist()
            out.extend(logits[a:b].tolist() for a, b in zip(pointers[:-1], pointers[1:]))
        return out

    def save(self, path):
        """One atomic checkpoint includes weights AND graph/model settings."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({
            "feature_version": FEATURE_VERSION,
            "model_config": asdict(self.model.config), "graph_config": asdict(self.graph_config),
            "encoder": self.emb.model,
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
        }, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path, emb, **kwargs):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["feature_version"] != FEATURE_VERSION or saved["encoder"] != emb.model:
            raise ValueError("Checkpoint feature schema or embedding encoder differs")
        model = PassageGAT(ModelConfig(**saved["model_config"]))
        model.load_state_dict(saved["state_dict"])
        return cls(model, emb, GraphConfig(**saved["graph_config"]), **kwargs)
