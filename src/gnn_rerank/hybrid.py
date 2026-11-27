"""Same passage graph/GAT as the GNN, with one frozen cross-encoder feature.

Only the extra node feature differs: [passage vector, query cosine, CE logit].
Neither supporting labels nor CE scores are used to select graph edges.
"""

from dataclasses import asdict
from pathlib import Path

import torch
from torch_geometric.data import Batch

from .gnn import ModelConfig, PassageGAT
from .graph import GraphConfig, build_graph

FEATURE_VERSION = 1


def build_hybrid_graph(row, emb, scores, config=None, *, with_labels=False):
    graph = build_graph(row, emb, config, with_labels=with_labels)
    # Use a raw logit, just as exported by the cross-encoder. No threshold or
    # gold annotation is added. Standard BGE inputs grow from 769 to 770 values.
    feature = torch.tensor(scores.get(row), dtype=torch.float32).unsqueeze(1)
    graph.x = torch.cat([graph.x, feature], dim=1)
    return graph


class HybridDataset:
    def __init__(self, rows, emb, scores, config=None):
        self.rows, self.emb, self.scores = rows, emb, scores
        self.config = config or GraphConfig()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return build_hybrid_graph(self.rows[index], self.emb, self.scores, self.config, with_labels=True)


class HybridReranker:
    name = 'hybrid'

    def __init__(self, model, emb, scores, graph_config=None, *, device=None, batch_size=64):
        if model.config.input_dim != emb.questions.shape[1] + 2:
            raise ValueError('Hybrid input must contain passage embedding + cosine + CE logit')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.emb, self.scores = emb, scores
        self.graph_config = graph_config or GraphConfig()
        self.batch_size = batch_size

    def score(self, rows):
        self.model.eval()
        output = []
        for start in range(0, len(rows), self.batch_size):
            graphs = [build_hybrid_graph(row, self.emb, self.scores, self.graph_config)
                      for row in rows[start:start + self.batch_size]]
            batch = Batch.from_data_list(graphs).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch.x, batch.edge_index).float().cpu()
            pointers = batch.ptr.cpu().tolist()
            output.extend(logits[a:b].tolist() for a, b in zip(pointers[:-1], pointers[1:]))
        return output

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save({
            'kind': 'hybrid', 'feature_version': FEATURE_VERSION,
            'model_config': asdict(self.model.config), 'graph_config': asdict(self.graph_config),
            'encoder': self.emb.model, 'cross_encoder_source': self.scores.source,
            'state_dict': {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
        }, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, emb, scores, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('kind') != 'hybrid' or saved['feature_version'] != FEATURE_VERSION:
            raise ValueError('Expected a compatible hybrid checkpoint')
        if saved['encoder'] != emb.model or saved['cross_encoder_source'] != scores.source:
            raise ValueError('Hybrid checkpoint requires the same BGE and cross-encoder sources')
        model = PassageGAT(ModelConfig(**saved['model_config']))
        model.load_state_dict(saved['state_dict'])
        return cls(model, emb, scores, GraphConfig(**saved['graph_config']), **kwargs)
