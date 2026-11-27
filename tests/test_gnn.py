"""Real CPU GAT inference/training and checkpoint round-trip checks."""

import copy
import json

import pytest
import torch

from gnn_rerank import eval as ev, gnn, train_gnn
from gnn_rerank.graph import GraphConfig, build_graph


def test_scoring_batching_and_checkpoint(graph_examples, tmp_path):
    rows, emb = graph_examples
    train_gnn.seed_everything(682)
    model = gnn.PassageGAT(gnn.ModelConfig(9, hidden_dim=8, heads=2, dropout=.2))
    scorer = gnn.GNNReranker(model, emb, GraphConfig(1), device='cpu')
    scores = scorer.score(rows)
    assert [len(s) for s in scores] == [6, 3]
    assert scorer.score([]) == []
    for i in range(2):
        assert scorer.score([rows[i]])[0] == pytest.approx(scores[i], abs=1e-6)
    unlabelled = copy.deepcopy(rows)
    for row in unlabelled:
        for p in row['passages']:
            del p['is_supporting']
    assert scorer.score(unlabelled) == scores
    assert ev.evaluate(scorer, rows)['n'] == 2
    scorer.save(tmp_path / 'best.pt')
    reloaded = gnn.GNNReranker.load(tmp_path / 'best.pt', emb, device='cpu')
    assert reloaded.graph_config == GraphConfig(1)
    for actual, expected in zip(reloaded.score(rows), scores):
        assert actual == pytest.approx(expected, abs=1e-6)
    emb.model = 'different'
    with pytest.raises(ValueError, match='encoder'):
        gnn.GNNReranker.load(tmp_path / 'best.pt', emb, device='cpu')


def test_actual_training_and_reuse(graph_examples, tmp_path):
    rows, emb = graph_examples
    config = train_gnn.TrainConfig(epochs=2, batch_size=2)
    model_config = gnn.ModelConfig(9, hidden_dim=8, heads=2, dropout=0)
    train_gnn.seed_everything(config.seed)
    initial = gnn.PassageGAT(model_config).state_dict()
    run = tmp_path / 'run'
    history = train_gnn.train(rows, rows, emb, emb, run, model_config=model_config, config=config, device='cpu')
    assert len(history) == 2
    assert all(record['val']['n'] == 2 for record in history)
    trained = gnn.GNNReranker.load(run / 'best.pt', emb, device='cpu')
    assert any(not torch.equal(initial[k], trained.model.state_dict()[k]) for k in initial)
    assert train_gnn.train(rows, rows, emb, emb, run, model_config=model_config, config=config, device='cpu') == history
    changed = copy.deepcopy(rows)
    changed[0]['passages'][0]['is_supporting'] = False
    with pytest.raises(ValueError, match='different data/settings'):
        train_gnn.train(changed, rows, emb, emb, run, model_config=model_config, config=config, device='cpu')


def test_best_epoch_not_last(graph_examples, tmp_path, monkeypatch):
    rows, emb = graph_examples
    observed = []

    def validate(scorer, supplied_rows):
        assert supplied_rows is rows
        observed.append({k: v.detach().clone() for k, v in scorer.model.state_dict().items()})
        value = 1.0 if len(observed) == 1 else 0.0
        return {'all_gold@5': value, 'recall@5': value}

    monkeypatch.setattr(train_gnn.ev, 'evaluate', validate)
    run = tmp_path / 'best'
    train_gnn.train(rows, rows, emb, emb, run, config=train_gnn.TrainConfig(epochs=2),
              model_config=gnn.ModelConfig(9, hidden_dim=8, heads=1), device='cpu')
    saved = gnn.GNNReranker.load(run / 'best.pt', emb, device='cpu').model.state_dict()
    assert all(torch.equal(saved[k], observed[0][k]) for k in saved)
    assert any(not torch.equal(saved[k], observed[1][k]) for k in saved)
    assert json.loads((run / 'complete.json').read_text())['best_all_gold@5'] == 1.


def test_tiny_overfit_without_message_passing(graph_examples):
    rows, emb = graph_examples
    train_gnn.seed_everything(682)
    g = build_graph(rows[0], emb, GraphConfig(0), with_labels=True)
    model = gnn.PassageGAT(gnn.ModelConfig(9, hidden_dim=8, heads=2, dropout=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    initial = loss_fn(model(g.x, g.edge_index), g.y).item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(g.x, g.edge_index), g.y)
        loss.backward()
        optimizer.step()
    logits = model(g.x, g.edge_index)
    assert loss_fn(logits, g.y).item() < initial * .1
    assert torch.equal(logits > 0, g.y.bool())
