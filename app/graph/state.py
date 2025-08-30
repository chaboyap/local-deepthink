# app/graph/state.py

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.documents import Document
from app.api.schemas import GraphRunParams
from app.core.config import ProviderConfig
from app.core.context import ServiceContext

class GraphState(TypedDict):
    """
    Represents the STATE of the graph. It is now fully serializable.
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
    all_rag_documents: List[Document]
    academic_papers: Optional[dict]
    is_code_request: bool
    session_id: str
    chat_history: List[dict]
    
    # Live services, not serialized
    services: ServiceContext
    
    # LLM and Config objects
    llm_config: ProviderConfig
    synthesizer_llm_config: ProviderConfig
    summarizer_llm_config: ProviderConfig
    embeddings_config: ProviderConfig