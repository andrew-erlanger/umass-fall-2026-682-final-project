"""Verified CE prediction caching, with no external model downloads."""

import copy
import json

import pytest
import torch

from gnn_rerank import hybrid_scores as hs


def test_cache_source_alignment_integrity_and_freezing(graph_examples, tmp_path, monkeypatch):
    rows, _ = graph_examples
    checkpoint = tmp_path / 'ce' / 'best'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text('{}')
    weights = checkpoint / 'model.safetensors'
    weights.write_bytes(b'test-checkpoint-v1')
    source = hs.checkpoint_hash(checkpoint)
    calls = []
    models = []

    class Scorer:
        name = 'cross_encoder'

        def __init__(self, *args, **kwargs):
            self.model = torch.nn.Linear(1, 1)
            models.append(self.model)
            calls.append(True)

        def score(self, batch):
            assert not any(p.requires_grad for p in self.model.parameters())
            return [[float(i) for i in range(len(row['passages']))] for row in batch]

    monkeypatch.setattr(hs.cross_encoder, 'CrossEncoderReranker', Scorer)
    path = tmp_path / 'features' / 'train.jsonl'
    actual = hs.cached_scores(rows, checkpoint, path, device='cpu')
    assert actual.source == source
    assert actual.get(rows[0]) == [0., 1., 2., 3., 4., 5.]
    assert hs.cached_scores(rows, checkpoint, path).signature() == actual.signature()
    assert len(calls) == 1
    # Gold labels are excluded from feature identity.
    relabeled = copy.deepcopy(rows)
    relabeled[0]['passages'][0]['is_supporting'] = False
    hs.cached_scores(relabeled, checkpoint, path)
    assert len(calls) == 1
    # Same question IDs but different order must never silently reuse features.
    reordered = copy.deepcopy(rows)
    reordered[0]['passages'].reverse()
    with pytest.raises(ValueError, match='passage inputs'):
        hs.load_scores(reordered, path, source=source)
    hs.cached_scores(reordered, checkpoint, path)
    assert len(calls) == 2
    weights.write_bytes(b'test-checkpoint-v2')
    with pytest.raises(ValueError, match='different source'):
        hs.load_scores(reordered, path, source=hs.checkpoint_hash(checkpoint))
    hs.cached_scores(reordered, checkpoint, path)
    assert len(calls) == 3
    path.write_text(path.read_text().replace('0.0', '9.0', 1))
    hs.cached_scores(reordered, checkpoint, path)
    assert len(calls) == 4
    assert weights.read_bytes() == b'test-checkpoint-v2'  # Export never writes model weights.
    assert json.loads(path.with_suffix('.meta.json').read_text())['scores_sha256'] == hs.file_hash(path)


def test_checkpoint_requires_model_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        hs.checkpoint_hash(tmp_path)
