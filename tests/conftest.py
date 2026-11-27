"""Small synthetic text/embedding fixtures; no network or model downloads."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class Rows(list):
    """Support the column lookup used by Hugging Face Dataset in cache code."""

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


@pytest.fixture
def graph_examples():
    import torch
    from gnn_rerank.graph import GraphEmbeddings, row_hash

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(682)
    rows = Rows([
        {'question_id': f'q{i}', 'question': f'question {i}', 'passages': [
            {'doc_id': f'q{i}_{j}', 'title': f'title {j}', 'text': f'text {i} {j}', 'is_supporting': j < 2}
            for j in range(size)
        ]}
        for i, size in enumerate((6, 3))
    ])
    emb = GraphEmbeddings(
        model='synthetic', question_ids=rows['question_id'],
        questions=torch.nn.functional.normalize(torch.randn(2, 8, generator=generator), dim=-1),
        passages=torch.nn.functional.normalize(torch.randn(9, 8, generator=generator), dim=-1),
        offsets=torch.tensor([0, 6, 9]), row_hashes=[row_hash(row) for row in rows],
    )
    return rows, emb
