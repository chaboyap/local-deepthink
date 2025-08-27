# app/services/llm_providers/ollama_provider.py

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from .base import LLMProvider

class OllamaProvider(LLMProvider):
    """Concrete implementation for the Ollama provider."""

    def get_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        # Provider only has ONE set of general-purpose defaults.
        defaults = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        }
        
        # User-provided kwargs will always override defaults
        params = {**defaults, **kwargs}
        return ChatOllama(model=model_name, **params)

    def get_embeddings_model(self, model_name: str, **kwargs) -> Embeddings:
        return OllamaEmbeddings(model=model_name, **kwargs)