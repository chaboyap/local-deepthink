# app/core/serialization.py
import json
from langchain_core.documents import Document
from app.api.schemas import GraphRunParams
from pydantic import BaseModel  # Make sure this import is present

def _default_serializer(obj):
    """Custom JSON serializer for our specific objects."""
    # LangChain's Document behaves like a Pydantic model for serialization.
    # This single check will correctly handle Document, GraphRunParams, ProviderConfig,
    # and any other Pydantic models by using the modern .model_dump() method.
    if isinstance(obj, (BaseModel, Document)):
        return obj.model_dump()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def serialize_state(state: dict) -> str:
    """
    [ARCHITECTURAL_ROLE]: Serialization Boundary Controller
    [PRIMARY_TASK]: Prepares the live GraphState for persistence by removing
                  non-serializable runtime objects (e.g., ServiceContext).
    [MECHANISM]: Creates a copy and uses .pop() to ensure the live, in-memory
                 state object is not mutated during serialization.
    """
    # Create a copy to prevent modifying the live state object during a run.
    state_to_serialize = state.copy()
    
    # Exclude the non-serializable ServiceContext before saving to Redis/file.
    state_to_serialize.pop('services', None)
    
    return json.dumps(state_to_serialize, default=_default_serializer)

def deserialize_state(json_str: str) -> dict:
    """Deserializes a JSON string back into a GraphState dictionary."""
    data = json.loads(json_str)
    
    # Re-hydrate special objects
    if 'all_rag_documents' in data and data['all_rag_documents']:
        data['all_rag_documents'] = [Document(**doc) for doc in data['all_rag_documents']]
    
    if 'params' in data and data['params']:
        data['params'] = GraphRunParams(**data['params'])
        
    return data