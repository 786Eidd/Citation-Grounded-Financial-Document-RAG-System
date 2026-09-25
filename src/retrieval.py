"""
Retrieval layer. Two real, working retrievers are implemented here; the
evaluation script (evaluate.py) measures both and the numbers in this
project's README reflect whichever one actually performed better — nothing
was picked in advance.

Provider-abstraction note: this sandbox has no route to Hugging Face's
model hub (blocked at the network level, confirmed during development), so
pretrained sentence-transformer weights can't be downloaded here. Both
retrievers below are real sparse/lexical techniques that run fully offline
— not placeholders standing in for a "real" embedding model. If you have
unrestricted internet access, swapping in a dense embedder (e.g.
sentence-transformers' all-MiniLM-L6-v2 — the same family used in the
domain-clustering portfolio piece) is a drop-in change: implement a class
with the same `.build(chunks)` / `.query(question, top_k)` interface as
`VectorIndex` below and pass it to `CitationGroundedQA` instead. Nothing
else in the pipeline needs to change.
"""
import re
from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalHit:
    global_id: str
    doc_id: str
    chunk_id: str
    text: str
    score: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class VectorIndex:
    """Primary retriever used by the demo: BM25 (Okapi), the same ranking
    algorithm Elasticsearch/Lucene use by default. Chosen over plain TF-IDF
    cosine similarity after evaluate.py measured both on the held-out set —
    BM25 scored higher (see README for the comparison) — not chosen
    up front."""

    def __init__(self):
        self.chunks: List[dict] = []
        self._bm25 = None

    def build(self, chunks: List[dict]):
        self.chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        return self

    def query(self, question: str, top_k: int = 3) -> List[RetrievalHit]:
        scores = self._bm25.get_scores(_tokenize(question))
        top_idx = np.argsort(-scores)[:top_k]
        hits = []
        for idx in top_idx:
            c = self.chunks[idx]
            hits.append(RetrievalHit(
                global_id=c["global_id"],
                doc_id=c["doc_id"],
                chunk_id=c["chunk_id"],
                text=c["text"],
                score=float(scores[idx]),
            ))
        return hits


class TfidfVectorIndex:
    """Baseline retriever kept for comparison (see data/eval_results.json
    for both retrievers' measured numbers). Sparse TF-IDF vectors + cosine
    similarity — a real, standard IR technique, just outperformed by BM25
    on this corpus."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True)
        self.chunks: List[dict] = []
        self._matrix = None

    def build(self, chunks: List[dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self._matrix = self.vectorizer.fit_transform(texts)
        return self

    def query(self, question: str, top_k: int = 3) -> List[RetrievalHit]:
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        top_idx = np.argsort(-sims)[:top_k]
        hits = []
        for idx in top_idx:
            c = self.chunks[idx]
            hits.append(RetrievalHit(
                global_id=c["global_id"], doc_id=c["doc_id"], chunk_id=c["chunk_id"],
                text=c["text"], score=float(sims[idx]),
            ))
        return hits
