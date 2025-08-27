# app/services/llm_providers/kobold_provider.py

from langchain_community.llms.koboldai import KoboldApiLLM 
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.chat_models.base import SimpleChatModel # Wrapper is still required
from .base import LLMProvider

class KoboldProvider(LLMProvider):
    """Concrete implementation for the KoboldAI API provider."""

    def get_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        endpoint = kwargs.pop("endpoint", "http://localhost:5001/api")
        
        params = {
            "temperature": 0.7,
            **kwargs
        }
        
        #  Instantiate the base LLM from the direct import
        #  Note the class name is KoboldApiLLM
        llm = KoboldApiLLM(endpoint=endpoint, **params)
        
        # Wrap the LLM to make it a compatible ChatModel
        return SimpleChatModel(llm=llm)

    def get_embeddings_model(self, model_name: str, **kwargs) -> Embeddings:
        raise NotImplementedError(
            "KoboldAI does not support embeddings via this integration. "
            "Please configure a different provider (e.g., ollama or gemini) for embeddings in your config.yaml."
        )