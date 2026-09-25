"""
FastAPI service exposing the citation-grounded QA engine.

Run with:
    uvicorn app:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "what was Visa's total volume in 2008?"}'
"""
import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from retrieval import VectorIndex
from qa_engine import CitationGroundedQA, REFUSAL_THRESHOLD

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

app = FastAPI(
    title="Citation-Grounded Financial Document RAG",
    description=(
        "Answers questions against a corpus of real SEC 10-K/10-Q filing "
        "excerpts, always returning a traceable citation, and refusing to "
        "answer when retrieval confidence is below a measured threshold "
        "rather than guessing."
    ),
)

_engine: CitationGroundedQA = None


@app.on_event("startup")
def load_index():
    global _engine
    with open(DATA_DIR / "corpus.json") as f:
        corpus = json.load(f)
    index = VectorIndex().build(corpus["chunks"])
    _engine = CitationGroundedQA(index, threshold=REFUSAL_THRESHOLD, top_k=3)


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float


class AskResponse(BaseModel):
    question: str
    status: str  # "cited" | "refused"
    citations: list[CitationOut]
    top_score: float


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = _engine.ask(req.question)
    return AskResponse(
        question=result.question,
        status=result.status,
        citations=[
            CitationOut(doc_id=c.doc_id, chunk_id=c.chunk_id, text=c.text, score=c.score)
            for c in result.citations
        ],
        top_score=result.top_score,
    )


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(_engine.index.chunks) if _engine else 0}
