"""
Citation-grounded QA engine: retrieval + citation objects + refusal logic.

Design goal (from the portfolio brief): answer financial-document questions
with citations traceable to the exact source passage, and refuse to answer
rather than hallucinate when retrieval evidence is weak.

This module doesn't try to compute the final numeric answer (that would
need an LLM or a program executor this environment doesn't have API access
to) — it does the part a citation-grounded RAG system is actually
responsible for: finding the right passage and proving where it came from.
A generation step could sit on top of this and synthesize a sentence from
the cited passages; it isn't included here because that layer isn't
independently testable without an LLM API key, and this build's whole point
was to produce numbers that are real.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from retrieval import RetrievalHit

# BM25 score below this -> insufficient evidence. Value comes from
# evaluate.py: the 80th percentile of BM25 scores measured on 15
# out-of-domain probe questions, run against the actual indexed corpus (see
# data/eval_results.json for the measured run this number came from).
REFUSAL_THRESHOLD = 14.384


@dataclass
class Citation:
    doc_id: str
    chunk_id: str
    text: str
    score: float


@dataclass
class QAResult:
    question: str
    status: str  # "cited" | "refused"
    citations: List[Citation] = field(default_factory=list)
    top_score: float = 0.0


class CitationGroundedQA:
    """`index` is any retriever exposing .build(chunks) / .query(question, top_k)
    — VectorIndex (BM25) or TfidfVectorIndex from retrieval.py, or a custom
    dense-embedding retriever with the same interface."""

    def __init__(self, index, threshold: float = REFUSAL_THRESHOLD, top_k: int = 3):
        self.index = index
        self.threshold = threshold
        self.top_k = top_k

    def ask(self, question: str) -> QAResult:
        hits: List[RetrievalHit] = self.index.query(question, top_k=self.top_k)
        top_score = hits[0].score if hits else 0.0

        if not hits or top_score < self.threshold:
            return QAResult(question=question, status="refused", citations=[], top_score=top_score)

        citations = [
            Citation(doc_id=h.doc_id, chunk_id=h.chunk_id, text=h.text, score=h.score)
            for h in hits
        ]
        return QAResult(question=question, status="cited", citations=citations, top_score=top_score)
