"""
FastAPI  entry point for the Amazon Reviews RAG system (Ollama-based).

This API exposes endpoints to:
    Accept user questions related to Amazon product reviews
    Generate answers using RAG pipeline
    Perform health checks to verify the service is running

The RAG logic is inside `rag_engine_ollama.generate_answer`.
"""


from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_engine_ollama import generate_answer

app = FastAPI(title="Amazon Reviews RAG API (Ollama)")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):
    return generate_answer(payload.question, k=5)

@app.get("/health")
def health():
    return {"status": "ok"}
