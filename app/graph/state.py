# app/graph/state.py

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.documents import Document
from app.rag.raptor import RAPTOR
from app.api.schemas import GraphRunParams
from app.core.config import ProviderConfig

class GraphState(TypedDict):
    """
    Represents the state of the graph. It's passed between nodes and updated
    at each step of the execution.
    """
    modules: List[dict]
    synthesis_context_queue: List[str]
    synthesis_execution_success: bool
    agent_personas: dict
    previous_solution: str
    current_problem: str
    original_request: str
    decomposed_problems: Dict[str, str]
    layers: List[dict]
    epoch: int
    max_epochs: int
    params: GraphRunParams
    all_layers_prompts: List[List[str]]
    agent_outputs: Annotated[dict, lambda a, b: {**a, **b}]
    memory: Annotated[dict, lambda a, b: {**a, **b}]
    critiques: Annotated[dict, lambda a, b: {**a, **b}]
    final_solution: dict
    perplexity_history: List[float]
    raptor_index: Optional[RAPTOR]
    all_rag_documents: List[Document]
    academic_papers: Optional[dict]
    is_code_request: bool
    session_id: str
    chat_history: List[dict]
    
    # LLM and Config objects
    llm: Any
    llm_config: ProviderConfig
    synthesizer_llm: Any
    synthesizer_llm_config: ProviderConfig
    summarizer_llm: Any
    summarizer_llm_config: ProviderConfig
    embeddings_model: Any
    embeddings_config: ProviderConfig