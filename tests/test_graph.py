"""Graph topology, label isolation, batching and embedding-cache alignment."""

import copy

import pytest
import torch
from torch_geometric.data import Batch

from gnn_rerank import embed, graph
from gnn_rerank.graph import GraphConfig, build_graph, knn_edges


def test_neighbor_topology_and_small_graphs():
    # Node 0's best neighbor is 1; node 2 also chooses 1.
    vectors = torch.tensor([[1., 0.], [.8, .6], [0., 1.]])
    edges = knn_edges(vectors, 1)
    assert set(map(tuple, edges.T.tolist())) == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert knn_edges(vectors, 99).shape == (2, 6)  # clamp k to n-1
    assert knn_edges(vectors[:1], 3).shape == (2, 0)
    assert knn_edges(vectors, 0).shape == (2, 0)
    with pytest.raises(ValueError):
        knn_edges(vectors, -1)
    # Equal similarities have a stable position-based tie break.
    tied = torch.ones(4, 2)
    assert torch.equal(knn_edges(tied, 1), knn_edges(tied, 1))


def test_features_labels_and_batch_isolation(graph_examples):
    rows, emb = graph_examples
    graph = build_graph(rows[0], emb, with_labels=True)
    assert graph.x.shape == (6, 9)
    q, passages = emb.get('q0')
    assert torch.allclose(graph.x[:, :8], passages.float())
    assert torch.allclose(graph.x[:, -1], passages @ q, atol=1e-6)
    assert graph.y.tolist() == [1., 1., 0., 0., 0., 0.]
    unlabelled = copy.deepcopy(rows[0])
    for passage in unlabelled['passages']:
        del passage['is_supporting']
    prediction_graph = build_graph(unlabelled, emb)
    assert prediction_graph.y is None
    assert torch.equal(graph.x, prediction_graph.x)
    assert torch.equal(graph.edge_index, prediction_graph.edge_index)
    other = build_graph(rows[1], emb)
    batch = Batch.from_data_list([prediction_graph, other])
    a, b = batch.edge_index
    assert torch.equal(batch.batch[a], batch.batch[b])  # no cross-question edges
    assert batch.ptr.tolist() == [0, 6, 9]


def test_alignment_rejects_changed_order_text_or_legacy(graph_examples):
    rows, emb = graph_examples
    for change in ('order', 'text'):
        changed = copy.deepcopy(rows[0])
        if change == 'order':
            changed['passages'].reverse()
        else:
            changed['question'] = 'different question'
        with pytest.raises(ValueError, match='content/order'):
            build_graph(changed, emb)
    emb.row_hashes = None
    with pytest.raises(ValueError, match='cached_embeddings'):
        build_graph(rows[0], emb)


def test_gnn_cache_preserves_baseline_and_checks_inputs(graph_examples, tmp_path, monkeypatch):
    rows, emb = graph_examples
    path = tmp_path / 'dev.pt'
    # Simulate the existing baseline file: it has no GNN-specific fingerprints.
    baseline = embed.Embeddings(emb.model, emb.question_ids, emb.questions, emb.passages, emb.offsets)
    baseline.save(path)
    original_bytes = path.read_bytes()
    calls = []

    def recompute(ds, model, **kwargs):
        calls.append(True)
        return baseline

    monkeypatch.setattr(embed, 'embed_split', recompute)
    verified = graph.cached_embeddings(rows, path, model='synthetic')
    assert verified.row_hashes == [graph.row_hash(row) for row in rows]
    assert (tmp_path / 'gnn' / 'dev.pt').exists()
    assert len(calls) == 1
    graph.cached_embeddings(rows, path, model='synthetic')
    assert len(calls) == 1  # Reuse matching GNN cache.
    changed = copy.deepcopy(rows)
    changed[0]['passages'][0]['text'] += ' changed'
    graph.cached_embeddings(changed, path, model='synthetic')
    assert len(calls) == 2
    assert path.read_bytes() == original_bytes  # Never overwrite shared baseline vectors.


def test_reuse_verified_shared_vectors_without_reencoding(graph_examples, tmp_path, monkeypatch):
    rows, emb = graph_examples
    path = tmp_path / 'dev.pt'
    emb.save(path)
    original_bytes = path.read_bytes()

    def unexpected_download(*args, **kwargs):
        raise AssertionError('Verified shared cache should be reused')

    monkeypatch.setattr(embed, 'embed_split', unexpected_download)
    actual = graph.cached_embeddings(rows, path, model='synthetic')
    assert torch.equal(actual.passages, emb.passages)
    assert not (tmp_path / 'gnn').exists()
    assert path.read_bytes() == original_bytes
