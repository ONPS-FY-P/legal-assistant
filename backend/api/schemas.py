"""
schemas.py

Purpose: Pydantic models define the exact shape of API requests/responses.
FastAPI uses these for automatic validation AND auto-generated docs
(visible at /docs once the server is running) -- so the contract
between backend and the future mobile app is explicit and self-documenting.
"""

from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str


class SourceUsed(BaseModel):
    article_number: str | None = None
    schedule_name: str | None = None
    title: str | None = None
    dense_score: float | None = None
    bm25_score: float | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources_used: list[SourceUsed]
    confidence_gate_triggered: bool
    top_dense_score: float | None = None
    top_bm25_score: float | None = None