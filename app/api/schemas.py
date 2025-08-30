# app/api/schemas.py

from pydantic import BaseModel, Field
from typing import List

class GraphRunParams(BaseModel):
    """Defines the user-configurable parameters for a graph run."""
    prompt: str = Field(..., min_length=10)
    mbti_archetypes: List[str] = Field(..., min_items=2)
    cot_trace_depth: int = Field(default=2, ge=2, le=16)
    num_questions: int = Field(default=10, ge=5, le=50, alias="num_questions_for_harvest") # Use alias for backend consistency
    num_epochs: int = Field(default=1, ge=1)
    vector_word_size: int = Field(default=2, ge=2, le=10)
    prompt_alignment: float = Field(default=1.0, ge=0.1, le=2.0)
    density: float = Field(default=1.0, ge=0.1, le=2.0)
    debug_mode: bool = False
    coder_debug_mode: bool = False
    
class GraphRunPayload(BaseModel):
    """The main payload for the /build_and_run_graph endpoint."""
    params: GraphRunParams