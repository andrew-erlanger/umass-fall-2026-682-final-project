"""Hybrid features, actual GAT training, and checkpoint/source compatibility."""

import copy
import json

import pytest
import torch

from gnn_rerank import eval as ev, gnn, hybrid, train_gnn, train_hybrid
from gnn_rerank.graph import GraphConfig, build_graph, row_hash
from gnn_rerank.hybrid_scores import CrossEncoderScores


def features(rows, source='frozen-ce'):
    return CrossEncoderScores(
        {row['question_id']: [float(j) for j in range(len(row['passages']))] for row in rows},
        {row['question_id']: row_hash(row) for row in rows}, source,
    )


def test_feature_append_does_not_change_graph_or_use_labels(graph_examples):
    rows, emb = graph_examples
    scores = features(rows)
    base = build_graph(rows[0], emb, GraphConfig(1), with_labels=True)
    data = hybrid.build_hybrid_graph(rows[0], emb, scores, GraphConfig(1), with_labels=True)
    assert data.x.shape == (6, 10)
    assert torch.equal(data.x[:, :-1], base.x)
    assert torch.equal(data.edge_index, base.edge_index)
    assert torch.equal(data.y, base.y)
    assert data.x[:, -1].tolist() == scores.get(rows[0])
    changed_scores = copy.deepcopy(scores)
    changed_scores.scores['q0'].reverse()
    altered = hybrid.build_hybrid_graph(rows[0], emb, changed_scores, GraphConfig(1))
    assert torch.equal(altered.edge_index, data.edge_index)
    assert not torch.equal(altered.x[:, -1], data.x[:, -1])
    no_labels = copy.deepcopy(rows[0])
    for p in no_labels['passages']:
        del p['is_supporting']
    prediction = hybrid.build_hybrid_graph(no_labels, emb, scores, GraphConfig(1))
    assert prediction.y is None and torch.equal(prediction.x, data.x)
    changed_row = copy.deepcopy(rows[0])
    changed_row['passages'].reverse()
    with pytest.raises(ValueError, match='passage inputs'):
        scores.get(changed_row)
    scores.scores['q0'][0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        scores.get(rows[0])


def test_hybrid_batching_and_reload(graph_examples, tmp_path):
    rows, emb = graph_examples
    scores = features(rows)
    train_gnn.seed_everything(682)
    model = gnn.PassageGAT(gnn.ModelConfig(10, hidden_dim=8, heads=2))
    scorer = hybrid.HybridReranker(model, emb, scores, GraphConfig(1), device='cpu')
    predictions = scorer.score(rows)
    assert [len(values) for values in predictions] == [6, 3]
    assert scorer.score([]) == []
    for i in range(2):
        assert scorer.score([rows[i]])[0] == pytest.approx(predictions[i], abs=1e-6)
    assert ev.evaluate(scorer, rows)['n'] == 2
    scorer.save(tmp_path / 'best.pt')
    reloaded = hybrid.HybridReranker.load(tmp_path / 'best.pt', emb, scores, device='cpu')
    assert reloaded.graph_config == GraphConfig(1)
    for expected, actual in zip(predictions, reloaded.score(rows)):
        assert actual == pytest.approx(expected, abs=1e-6)
    with pytest.raises(ValueError, match='sources'):
        hybrid.HybridReranker.load(tmp_path / 'best.pt', emb, features(rows, source='other-ce'))
    with pytest.raises(ValueError, match='Hybrid input'):
        hybrid.HybridReranker(gnn.PassageGAT(gnn.ModelConfig(9)), emb, scores)


def test_training_updates_only_gat_and_reuses_matching_run(graph_examples, tmp_path):
    rows, emb = graph_examples
    scores = features(rows)
    original_scores = copy.deepcopy(scores)
    original_vectors = emb.passages.clone()
    config = train_gnn.TrainConfig(epochs=2, batch_size=2)
    model_config = gnn.ModelConfig(10, hidden_dim=8, heads=1)
    train_gnn.seed_everything(config.seed)
    initial = gnn.PassageGAT(model_config).state_dict()
    run = tmp_path / 'run'
    history = train_hybrid.train(rows, rows, emb, emb, scores, scores, run,
        model_config=model_config, config=config, device='cpu')
    saved = hybrid.HybridReranker.load(run / 'best.pt', emb, scores, device='cpu')
    assert any(not torch.equal(initial[k], saved.model.state_dict()[k]) for k in initial)
    assert torch.equal(original_vectors, emb.passages)
    assert scores == original_scores
    assert len(history) == 2 and history[0]['val']['n'] == 2
    assert train_hybrid.train(rows, rows, emb, emb, scores, scores, run,
        model_config=model_config, config=config, device='cpu') == history
    changed = features(rows)
    changed.scores['q0'][0] += 1.
    with pytest.raises(ValueError, match='different settings/data/CE scores'):
        train_hybrid.train(rows, rows, emb, emb, changed, scores, run,
            model_config=model_config, config=config, device='cpu')
    with pytest.raises(ValueError, match='same frozen'):
        train_hybrid.train(rows, rows, emb, emb, scores, features(rows, 'other'), tmp_path / 'other')


def test_hybrid_saves_best_epoch_not_last(graph_examples, tmp_path, monkeypatch):
    rows, emb = graph_examples
    scores = features(rows)
    observed = []
    val_rows = rows[:1]

    def validate(scorer, supplied_rows):
        assert supplied_rows is val_rows
        observed.append({key: value.detach().clone() for key, value in scorer.model.state_dict().items()})
        value = 1. if len(observed) == 1 else 0.
        return {'all_gold@5': value, 'recall@5': value}

    monkeypatch.setattr(train_hybrid.ev, 'evaluate', validate)
    run = tmp_path / 'run'
    train_hybrid.train(rows, val_rows, emb, emb, scores, scores, run,
        config=train_gnn.TrainConfig(epochs=2), model_config=gnn.ModelConfig(10, hidden_dim=8, heads=1), device='cpu')
    saved = hybrid.HybridReranker.load(run / 'best.pt', emb, scores, device='cpu').model.state_dict()
    assert all(torch.equal(saved[k], observed[0][k]) for k in saved)
    assert any(not torch.equal(saved[k], observed[1][k]) for k in saved)
    assert json.loads((run / 'complete.json').read_text())['best_all_gold@5'] == 1.
