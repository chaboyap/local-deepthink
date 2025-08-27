# app/services/llm_providers/__init__.py

from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .kobold_provider import KoboldProvider
from .native_gemini_provider import NativeGeminiProvider

PROVIDER_MAP = {
    "ollama": OllamaProvider,
    "gemini": NativeGeminiProvider,
    "kobold": KoboldProvider,
}

def get_provider(provider_name: str) -> LLMProvider:
    """
    Factory function to get an instance of an LLM provider.
    """
    provider_class = PROVIDER_MAP.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown LLM provider: '{provider_name}'. Available providers are: {list(PROVIDER_MAP.keys())}")
    return provider_class()