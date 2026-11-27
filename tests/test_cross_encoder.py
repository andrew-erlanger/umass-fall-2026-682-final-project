"""Offline checks: tiny local transformer, no model or dataset downloads."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn_rerank import cross_encoder as ce


def rows():
    return [
        {"question_id": "q1", "question": "where author born", "passages": [
            {"title": "book", "text": "author wrote book", "is_supporting": True},
            {"title": "author", "text": "born canada", "is_supporting": True},
            {"title": "other", "text": "unrelated", "is_supporting": False},
        ]},
        {"question_id": "q2", "question": "where book", "passages": [
            {"title": "book", "text": "canada", "is_supporting": False},
        ]},
    ]


def test_pair_boundaries_and_labels():
    pairs = ce.PairDataset(rows())
    assert len(pairs) == 4
    assert pairs[0] == {"question": "where author born", "passage": "book: author wrote book", "label": 1.0}
    assert pairs[2]["label"] == 0.0
    assert pairs[3]["question"] == "where book"
    with pytest.raises(IndexError):
        pairs[4]


@pytest.fixture
def tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    torch.manual_seed(682)
    torch.set_num_threads(1)
    vocab = {word: i for i, word in enumerate([
        "[UNK]", "[PAD]", "where", "author", "born", "book", "wrote", "canada", "other", "unrelated", ":",
    ])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    config = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=24, num_labels=1,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    path = tmp_path / "initial"
    transformers.BertForSequenceClassification(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_score_alignment_label_independence_and_reload(tiny_model, tmp_path):
    scorer = ce.CrossEncoderReranker(tiny_model, batch_size=2, max_length=32, device="cpu")
    examples = rows()
    scores = scorer.score(examples)
    assert list(map(len, scores)) == [3, 1]
    assert scorer.score([]) == []
    assert scorer.score([{"question": "q", "passages": []}]) == [[]]
    # Scoring must not require gold labels and must preserve passage order.
    reordered = {"question": examples[0]["question"], "passages": [
        {k: v for k, v in p.items() if k != "is_supporting"}
        for p in reversed(examples[0]["passages"])
    ]}
    assert scorer.score([reordered])[0] == pytest.approx(list(reversed(scores[0])), abs=1e-6)
    scorer.save(tmp_path / "saved")
    reloaded = ce.CrossEncoderReranker(tmp_path / "saved", device="cpu")
    assert reloaded.max_length == 32
    for actual, expected in zip(reloaded.score(examples), scores):
        assert actual == pytest.approx(expected, abs=1e-6)


def test_training_updates_weights_and_keeps_best_epoch(tiny_model, tmp_path, monkeypatch):
    import torch

    observed = []
    initial = ce.CrossEncoderReranker(tiny_model, device="cpu").model.state_dict()
    val = rows()[:1]

    def validate(reranker, supplied_rows):
        assert supplied_rows is val  # Never evaluate a different/test split while fitting.
        observed.append({k: v.detach().clone() for k, v in reranker.model.state_dict().items()})
        value = 1.0 if len(observed) == 1 else 0.0
        return {"all_gold@5": value, "recall@5": value, "n": 1}

    monkeypatch.setattr(ce.ev, "evaluate", validate)
    run = tmp_path / "run"
    history = ce.train(rows(), val, run, model=tiny_model, device="cpu", config=ce.TrainConfig(
        epochs=2, batch_size=2, learning_rate=1e-3, max_length=32,
    ))
    saved = ce.CrossEncoderReranker(run / "best", device="cpu").model.state_dict()
    assert len(history) == 2
    assert json.loads((run / "complete.json").read_text())["best_all_gold@5"] == 1.0
    assert any(not torch.equal(initial[k], observed[0][k]) for k in initial)
    assert all(torch.equal(saved[k], observed[0][k]) for k in saved)
    assert any(not torch.equal(saved[k], observed[1][k]) for k in saved)
    with pytest.raises(FileExistsError):
        ce.train(rows(), val, run, model=tiny_model, device="cpu")
