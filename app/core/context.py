# app/core/context.py

from typing import Any
from pydantic import BaseModel
from app.rag.raptor import RAPTOR

class ServiceContext(BaseModel):
    """Holds all non-serializable, heavy service clients for a session."""
    llm: Any
    synthesizer_llm: Any
    summarizer_llm: Any
    embeddings_model: Any
    raptor_index: RAPTOR | None = None

    class Config:
        arbitrary_types_allowed = True