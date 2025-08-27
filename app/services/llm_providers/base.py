# app/services/llm_providers/base.py

from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

class LLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    Defines the interface that all concrete provider implementations must follow.
    """
    
    @abstractmethod
    def get_chat_model(self, model_name: str, **kwargs) -> BaseChatModel:
        """
        Returns a LangChain-compatible chat model instance.

        Args:
            model_name: The specific model to initialize (e.g., 'qwen3:7b').
            **kwargs: Provider-specific parameters (temperature, api_key, etc.).

        Returns:
            An instance of a class that inherits from BaseChatModel.
        """
        pass

    @abstractmethod
    def get_embeddings_model(self, model_name: str, **kwargs) -> Embeddings:
        """
        Returns a LangChain-compatible embeddings model instance.

        Args:
            model_name: The specific embeddings model to initialize.
            **kwargs: Provider-specific parameters.

        Returns:
            An instance of a class that inherits from Embeddings.
        """
        pass