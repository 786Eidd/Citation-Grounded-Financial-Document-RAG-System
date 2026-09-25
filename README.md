# Citation-Grounded Financial Document RAG

Answers questions against real SEC filing excerpts, always returning a
traceable citation to the exact source passage — and refuses to answer# Citation-Grounded Financial Document RAG

**Author:** Eid Muhammad — AI Engineer
**Contact:** eiddmohammad786@gmail.com

Answers questions against real SEC filing excerpts, always returning a
traceable citation to the exact source passage — and refuses to answer
rather than guess when retrieval confidence is too low.

This was built to close a specific gap: my portfolio had embedding/retrieval
work (domain-clustering pipeline) and production-deployment work (LoRA
fine-tuned classifier) separately, but nothing that combined them into an
actual RAG system with citations. Every number in this README came from
running the code in `src/`, not from a template — see
[`data/eval_results.json`](data/eval_results.json) for the raw output.

## What it does

1. **Ingests** real financial filing excerpts and chunks them into citable
   passages, each tagged with its source document and exact location.
2. **Retrieves** the most relevant passages for a question using lexical
   search (BM25).
3. **Cites** every answer with the source document, the exact passage, and a
   confidence score — never a bare claim.
4. **Refuses** to answer when the best match is below a measured confidence
   threshold, instead of guessing.

## Data

[FinQA](https://github.com/czyssrs/FinQA) (Chen et al., 2021) — a public
research dataset of real excerpts from S&P 500 companies' actual SEC 10-K /
10-Q filings, each paired with a question and a human-annotated "gold"
citation (the exact sentence or table row the answer depends on). That gold
annotation is what makes it possible to measure real retrieval accuracy
below, instead of eyeballing it.

This build indexes **80 distinct filing excerpts** (randomly sampled,
seeded for reproducibility — see `src/ingest.py`), producing **2,241
citable chunks** (1,846 text passages + 395 table rows converted to
sentences).

## Architecture

```
PDF/filing text
      │
      ▼
  ingest.py  ──►  chunk + tag with (doc_id, chunk_id, type)
      │
      ▼
 retrieval.py ──►  BM25 index over all chunks
      │
      ▼
 qa_engine.py ──►  retrieve top-k, threshold check
      │                    │
   score ≥ threshold   score < threshold
      │                    │
      ▼                    ▼
  return citations     refuse ("insufficient evidence")
      │
      ▼
   app.py (FastAPI /ask endpoint)
```

## Why BM25, not a transformer embedding model

The environment this was built in has no route to Hugging Face's model hub
(confirmed blocked at the network level during development), so pretrained
sentence-transformer weights couldn't be downloaded. Rather than fake that
step, retrieval here runs on two real, standard lexical techniques —
**BM25** (Okapi, the same ranking algorithm Elasticsearch/Lucene use by
default) and **TF-IDF cosine similarity** — and `src/evaluate.py` measures
both on the held-out set before picking one:

| Retriever | Top-1 accuracy | Top-3 accuracy |
|---|---|---|
| **BM25** (used) | **41.25%** | **56.25%** |
| TF-IDF cosine | 32.5% | 50.0% |

`retrieval.py` defines the retriever as a swappable interface
(`.build(chunks)` / `.query(question, top_k)`) for exactly this reason — the
same pattern as the provider-abstraction layer in my outreach-agent
project. Dropping in `sentence-transformers` (all-MiniLM-L6-v2, the same
model family used in my domain-clustering portfolio piece) once model-hub
access is available is a same-interface swap, not a rewrite.

## Measured results

Full run in [`data/eval_results.json`](data/eval_results.json).

- **Pure retrieval accuracy** (does the top match contain the gold
  citation, refusal disabled): **41.25% top-1, 56.25% top-3**, on 80
  held-out questions against a corpus built from real SEC filings.
- **At the operating refusal threshold**: of 80 real questions, the system
  answered 72 and refused 8 as insufficient evidence; of the 72 it
  answered, 33 cited the correct passage (**45.8% precision when
  answering**).
- **Refusal accuracy on out-of-domain questions**: 80% (12 of 15 clearly
  unrelated questions — e.g. "What is the capital of Bhutan?" — correctly
  refused rather than answered with an unrelated citation).
- **Latency**: 4.4ms average, 8.2ms p95 per query, over BM25 search across
  2,241 chunks.

### Honest limitations

- FinQA's questions are numeric-reasoning questions ("what was the average
  X from 2015 to 2017") that frequently share little vocabulary with their
  own evidence row — that's *why* lexical retrieval alone caps out around
  40-55%, not a bug in the implementation. Dense embeddings would likely do
  better here precisely because they capture meaning, not just word
  overlap; that's the strongest argument for the swap-in path above.
- In-domain and out-of-domain confidence scores overlap (see
  `in_domain_score_stats` / `out_of_domain_score_stats` in the eval
  output), so no single threshold perfectly separates "answerable" from
  "not answerable" — the 80th-percentile threshold used here is a
  documented tradeoff, not a solved problem.
- This build stops at retrieval + citation, deliberately. A generation
  layer that synthesizes a fluent answer from the cited passage would need
  an LLM API this environment isn't authenticated against; adding one is a
  clearly scoped next step, not a hidden gap.

## Running it

```bash
pip install -r requirements.txt
python src/ingest.py      # rebuilds data/corpus.json from data/finqa_dev.json
python src/evaluate.py    # rebuilds data/eval_results.json
uvicorn src.app:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "what was the operating margin for 2012?"}'
```

## Stack

Python, BM25 (`rank_bm25`) / scikit-learn (TF-IDF baseline), FastAPI,
pydantic. No LLM API dependency — retrieval and citation only, by design
(see Honest limitations above).

## Data attribution

Corpus built from [FinQA](https://github.com/czyssrs/FinQA) (Chen et al.,
*"FinQA: A Dataset of Numerical Reasoning over Financial Data"*, EMNLP
2021), MIT licensed — see `data/FINQA_LICENSE.txt`. FinQA's excerpts are
themselves drawn from real, publicly filed SEC 10-K/10-Q documents.

rather than guess when retrieval confidence is too low.

This was built to close a specific gap: my portfolio had embedding/retrieval
work (domain-clustering pipeline) and production-deployment work (LoRA
fine-tuned classifier) separately, but nothing that combined them into an
actual RAG system with citations. Every number in this README came from
running the code in `src/`, not from a template — see
[`data/eval_results.json`](data/eval_results.json) for the raw output.

## What it does

1. **Ingests** real financial filing excerpts and chunks them into citable
   passages, each tagged with its source document and exact location.
2. **Retrieves** the most relevant passages for a question using lexical
   search (BM25).
3. **Cites** every answer with the source document, the exact passage, and a
   confidence score — never a bare claim.
4. **Refuses** to answer when the best match is below a measured confidence
   threshold, instead of guessing.

## Data

[FinQA](https://github.com/czyssrs/FinQA) (Chen et al., 2021) — a public
research dataset of real excerpts from S&P 500 companies' actual SEC 10-K /
10-Q filings, each paired with a question and a human-annotated "gold"
citation (the exact sentence or table row the answer depends on). That gold
annotation is what makes it possible to measure real retrieval accuracy
below, instead of eyeballing it.

This build indexes **80 distinct filing excerpts** (randomly sampled,
seeded for reproducibility — see `src/ingest.py`), producing **2,241
citable chunks** (1,846 text passages + 395 table rows converted to
sentences).

## Architecture

```
PDF/filing text
      │
      ▼
  ingest.py  ──►  chunk + tag with (doc_id, chunk_id, type)
      │
      ▼
 retrieval.py ──►  BM25 index over all chunks
      │
      ▼
 qa_engine.py ──►  retrieve top-k, threshold check
      │                    │
   score ≥ threshold   score < threshold
      │                    │
      ▼                    ▼
  return citations     refuse ("insufficient evidence")
      │
      ▼
   app.py (FastAPI /ask endpoint)
```

## Why BM25, not a transformer embedding model

The environment this was built in has no route to Hugging Face's model hub
(confirmed blocked at the network level during development), so pretrained
sentence-transformer weights couldn't be downloaded. Rather than fake that
step, retrieval here runs on two real, standard lexical techniques —
**BM25** (Okapi, the same ranking algorithm Elasticsearch/Lucene use by
default) and **TF-IDF cosine similarity** — and `src/evaluate.py` measures
both on the held-out set before picking one:

| Retriever | Top-1 accuracy | Top-3 accuracy |
|---|---|---|
| **BM25** (used) | **41.25%** | **56.25%** |
| TF-IDF cosine | 32.5% | 50.0% |

`retrieval.py` defines the retriever as a swappable interface
(`.build(chunks)` / `.query(question, top_k)`) for exactly this reason — the
same pattern as the provider-abstraction layer in my outreach-agent
project. Dropping in `sentence-transformers` (all-MiniLM-L6-v2, the same
model family used in my domain-clustering portfolio piece) once model-hub
access is available is a same-interface swap, not a rewrite.

## Measured results

Full run in [`data/eval_results.json`](data/eval_results.json).

- **Pure retrieval accuracy** (does the top match contain the gold
  citation, refusal disabled): **41.25% top-1, 56.25% top-3**, on 80
  held-out questions against a corpus built from real SEC filings.
- **At the operating refusal threshold**: of 80 real questions, the system
  answered 72 and refused 8 as insufficient evidence; of the 72 it
  answered, 33 cited the correct passage (**45.8% precision when
  answering**).
- **Refusal accuracy on out-of-domain questions**: 80% (12 of 15 clearly
  unrelated questions — e.g. "What is the capital of Bhutan?" — correctly
  refused rather than answered with an unrelated citation).
- **Latency**: 4.4ms average, 8.2ms p95 per query, over BM25 search across
  2,241 chunks.

### Honest limitations

- FinQA's questions are numeric-reasoning questions ("what was the average
  X from 2015 to 2017") that frequently share little vocabulary with their
  own evidence row — that's *why* lexical retrieval alone caps out around
  40-55%, not a bug in the implementation. Dense embeddings would likely do
  better here precisely because they capture meaning, not just word
  overlap; that's the strongest argument for the swap-in path above.
- In-domain and out-of-domain confidence scores overlap (see
  `in_domain_score_stats` / `out_of_domain_score_stats` in the eval
  output), so no single threshold perfectly separates "answerable" from
  "not answerable" — the 80th-percentile threshold used here is a
  documented tradeoff, not a solved problem.
- This build stops at retrieval + citation, deliberately. A generation
  layer that synthesizes a fluent answer from the cited passage would need
  an LLM API this environment isn't authenticated against; adding one is a
  clearly scoped next step, not a hidden gap.

## Running it

```bash
pip install -r requirements.txt
python src/ingest.py      # rebuilds data/corpus.json from data/finqa_dev.json
python src/evaluate.py    # rebuilds data/eval_results.json
uvicorn src.app:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "what was the operating margin for 2012?"}'
```

## Stack

Python, BM25 (`rank_bm25`) / scikit-learn (TF-IDF baseline), FastAPI,
pydantic. No LLM API dependency — retrieval and citation only, by design
(see Honest limitations above).

## Data attribution

Corpus built from [FinQA](https://github.com/czyssrs/FinQA) (Chen et al.,
*"FinQA: A Dataset of Numerical Reasoning over Financial Data"*, EMNLP
2021), MIT licensed — see `data/FINQA_LICENSE.txt`. FinQA's excerpts are
themselves drawn from real, publicly filed SEC 10-K/10-Q documents.
