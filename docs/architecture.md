# Architecture

Diagrams of the data model, the modules and the API. Signatures and details live in the docstrings.

## 1. Data ontology

```mermaid
classDiagram
    direction LR
    class DatasetDict {
        train
        val: 10% of train, seed 682
        dev: our test set
    }
    class Question {
        question_id: str
        question: str
        answer: str
        answer_aliases: list~str~
        gold_support_ids: list~str~
        dataset: str
        question_type: str
        hops: int
    }
    class Passage {
        doc_id: str
        title: str
        text: str
        is_supporting: bool
    }
    class Embeddings {
        model: str
        question_ids: list~str~
        questions: Tensor n_q x d
        passages: Tensor n_p x d
        offsets: Tensor n_q+1
        doc_ids: list~str~
        get(question_id) q, P
    }
    class ScoreRecord {
        question_id: str
        method: str
        doc_ids: list~str~
        scores: list~float~
    }
    class Report {
        version, dataset, limit
        min_gold, shuffle, seed
        sizes, dropped, stats per split
    }
    DatasetDict "1" *-- "many" Question : rows
    Question "1" *-- "10 or 20" Passage : passages, shuffled
    Report "1" -- "1" DatasetDict : describes
    Embeddings ..> Question : keyed by question_id
    Embeddings ..> Passage : row j = passage j, checked by doc_id
    ScoreRecord ..> Question : one line per question
    ScoreRecord ..> Passage : score j = passage j, checked by doc_id
```

| Field | Values |
| --- | --- |
| `doc_id` | `<question_id>_<original index>`. It survives the shuffle, so it identifies a passage everywhere |
| `dataset` | `hotpotqa` · `2wiki` · `musique` |
| `question_type` | hotpotqa: `bridge` / `comparison` · 2wiki: its `type` field · musique: `2hop` / `3hop` / `4hop` |
| `hops` | musique: from the id · others: number of gold passages |
| passages per question | hotpotqa 10 · 2wiki 10 · musique 20, minus empty and duplicate passages |

## 2. Module structure

```mermaid
flowchart TB
    subgraph nbs [notebooks]
        starter[starter.ipynb]
        baselines[baselines.ipynb]
    end

    subgraph pkg [src/gnn_rerank]
        m_data["data.py<br/>load · split_train_val"]
        m_pre["preprocess.py<br/>prepare · clean"]
        m_emb["embed.py<br/>Embeddings · cached"]
        m_rr["rerank.py<br/>Reranker · RandomReranker · CosineReranker"]
        m_eval["eval.py<br/>evaluate · evaluate_by · save/load_scores · results_table"]
        m_future["planned: cross-encoder · GNN · hybrid"]
    end

    subgraph drive [Google Drive: MyDrive/umass-fall-2026-682-final-project]
        d_prep[("prepared/{dataset}/<br/>data/ + report.json")]
        d_emb[("embeddings/{model}/{dataset}/{split}.pt")]
        d_scores[("results/scores/{method}_{dataset}_{split}.jsonl")]
    end

    hf[(Hugging Face Hub)]
    st[(sentence-transformers)]

    starter --> m_data
    baselines --> m_pre & m_emb & m_rr & m_eval
    m_pre --> m_data
    m_data --> hf
    m_emb --> st
    m_rr -.->|type only| m_emb
    m_future -.->|implements Reranker| m_rr
    m_eval -.->|calls any Reranker, duck-typed| m_rr

    m_pre <-->|read / write| d_prep
    m_emb <-->|read / write| d_emb
    m_eval <-->|save / load| d_scores
```

Solid arrows are imports. Dashed arrows are interface relationships.

### Caches and what invalidates them

| Cache | Written by | Reused only if | On mismatch |
| --- | --- | --- | --- |
| `prepared/<dataset>[_limitN]/` | `preprocess.prepare` | `version`, `min_gold`, `shuffle` and `seed` all match `report.json` | rebuild |
| `embeddings/<model>/<dataset>/<split>.pt` | `embed.cached` | same model, question ids **and passage `doc_id` order** | recompute |
| `results/scores/*.jsonl` | `eval.save_scores` | `eval.aligned` finds every question **with the same passage order** | raise |

Bump `preprocess.VERSION` whenever a prepared row's contents change.

## 3. API

### Pipeline

```mermaid
flowchart LR
    hub[(HF Hub)] -->|"preprocess.prepare(name, limit, cache_dir)"| ds["DatasetDict<br/>train · val · dev"]
    ds -->|"embed.cached(ds[split], path)"| emb[Embeddings]
    emb -->|"CosineReranker(emb)"| rr{{"Reranker<br/>.name<br/>.score(rows) → list[list[float]]"}}
    yours[your method] -.->|implements| rr
    rr -->|"eval.evaluate(rr, ds[split], scores_path=…)"| metrics["metrics<br/>dict[str, float] + n"]
    rr -->|"scores_path"| file[("scores .jsonl")]
    file -->|"eval.load_scores → eval.aligned(ds, saved)"| scores["list[list[float]]"]
    scores -->|"eval.evaluate_scores(rows, scores)"| metrics
    scores -->|"eval.evaluate_by(rows, scores, key)"| groups["dict[group, metrics]"]
    metrics -->|"eval.results_table({(method, dataset): metrics})"| table[markdown table]
```

### `eval.evaluate` and the `Reranker` contract

```mermaid
sequenceDiagram
    participant NB as notebook
    participant EV as eval.evaluate
    participant RR as Reranker
    participant QM as eval.question_metrics
    NB->>EV: evaluate(reranker, ds, scores_path)
    loop batches of 64 rows
        EV->>RR: score(rows)
        RR-->>EV: one list per row, one finite float per passage, in row["passages"] order
    end
    opt scores_path given
        EV->>EV: save_scores: question_id, method, doc_ids, scores
    end
    loop each question
        EV->>QM: question_metrics(row, scores)
        Note over QM: rank by score, ties by position.<br/>Raises on length mismatch, NaN/inf, or no gold
        QM-->>EV: recall@k, all_gold@k, r_precision, all_gold@G, mrr, ndcg@10
    end
    EV-->>NB: mean of each metric, plus n
```

| Metric | Per question | Use |
| --- | --- | --- |
| `recall@k` | gold in top k / G | k ∈ {1, 2, 5, 10} |
| `all_gold@k` | 1 if all G gold in top k | multi-hop answerability |
| `r_precision` | gold in top G / G | |
| `all_gold@G` | 1 if top G are exactly the gold | headline metric, fair across 2–4 hops |
| `mrr` | 1 / rank of first gold | |
| `ndcg@10` | binary relevance | |

G = that question's number of gold passages. Tune on `val`, report on `dev`.
