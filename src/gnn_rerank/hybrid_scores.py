"""Export frozen cross-encoder predictions with passage-order/source checks.

The old dev-only score files do not identify the checkpoint or passage text.
Hybrid features therefore use their own verified train/val/dev cache. Existing
cross-encoder files and checkpoints are read only.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from . import cross_encoder, eval as ev
from .graph import row_hash


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hash(path):
    """Identify model weights, tokenizer and saved inference settings together."""
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not (path / 'config.json').exists() or not any(p.suffix in ('.bin', '.safetensors') for p in files):
        raise FileNotFoundError(f'{path}: expected a saved cross-encoder checkpoint')
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file_hash(file).encode())
    return digest.hexdigest()


@dataclass
class CrossEncoderScores:
    scores: dict[str, list[float]]
    input_hashes: dict[str, str]
    source: str  # Fingerprint of the frozen cross-encoder checkpoint.

    def get(self, row):
        qid = row['question_id']
        if qid not in self.scores or self.input_hashes.get(qid) != row_hash(row):
            raise ValueError(f'{qid}: cross-encoder features have missing/different passage inputs')
        values = self.scores[qid]
        if len(values) != len(row['passages']) or not all(math.isfinite(v) for v in values):
            raise ValueError(f'{qid}: expected one finite cross-encoder logit per passage')
        return values

    def signature(self):
        """Run identity includes actual scores, not just their file location."""
        digest = hashlib.sha256(self.source.encode())
        for qid in sorted(self.scores):
            record = [qid, self.input_hashes[qid], self.scores[qid]]
            digest.update(json.dumps(record).encode())
        return digest.hexdigest()


def load_scores(rows, path, *, source):
    """Require a complete, unchanged cache from this checkpoint and these rows."""
    path = Path(path)
    metadata = json.loads(path.with_suffix('.meta.json').read_text())
    if metadata['source'] != source or metadata['scores_sha256'] != file_hash(path):
        raise ValueError('Cross-encoder score cache has a different source or modified scores')
    scores, hashes = {}, {}
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            qid = record['question_id']
            if qid in scores:
                raise ValueError(f'Duplicate question in cross-encoder scores: {qid}')
            scores[qid], hashes[qid] = record['scores'], record['input_hash']
    result = CrossEncoderScores(scores, hashes, source)
    if set(scores) != {row['question_id'] for row in rows}:
        raise ValueError('Cross-encoder cache contains a different question set')
    for row in rows:
        result.get(row)
    return result


def cached_scores(rows, checkpoint, path, *, source=None, batch_size=32, device=None):
    """Reuse verified predictions or run the frozen scorer, never fine-tune it."""
    path = Path(path)
    source = source or checkpoint_hash(checkpoint)
    if path.exists() and path.with_suffix('.meta.json').exists():
        try:
            return load_scores(rows, path, source=source)
        except (ValueError, KeyError):
            print(f'{path}: different/corrupt score cache; recomputing')
    print(f'Exporting frozen cross-encoder scores: {path}')
    scorer = cross_encoder.CrossEncoderReranker(checkpoint, batch_size=batch_size, device=device)
    # Inference already disables autograd; explicitly freeze weights as well.
    scorer.model.requires_grad_(False)
    values = ev.score_all(scorer, rows)
    result = CrossEncoderScores(
        {row['question_id']: scores for row, scores in zip(rows, values, strict=True)},
        {row['question_id']: row_hash(row) for row in rows}, source,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps({
                'question_id': row['question_id'], 'method': 'cross_encoder',
                'scores': result.get(row), 'input_hash': row_hash(row),
            }) + '\n')
    temporary.replace(path)
    # Written last: missing/mismatched metadata causes a safe rebuild next time.
    metadata = path.with_suffix('.meta.json')
    temporary = metadata.with_suffix('.tmp')
    temporary.write_text(json.dumps({'source': source, 'scores_sha256': file_hash(path)}, indent=2))
    temporary.replace(metadata)
    return result
