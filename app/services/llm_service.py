# app/services/llm_service.py

import traceback
from app.core.config import settings
from typing import List, Tuple
from typing import List, Tuple, cast
import logging
from pydantic import BaseModel

from google.genai.errors import ServerError
from google.api_core.exceptions import DeadlineExceeded

from app.services.llm_providers import get_provider
from langchain_core.embeddings import Embeddings
from app.utils.mock_llms import CoderMockLLM, MockLLM
from app.core.config import ProviderConfig, AppSettings
from langchain_core.language_models import BaseChatModel

from tenacity import stop_after_attempt, wait_exponential

class LLMInitializationParams(BaseModel):
    """Defines only the parameters needed to initialize LLMs."""
    debug_mode: bool = False
    coder_debug_mode: bool = False

def _create_chat_model(model_config: ProviderConfig) -> BaseChatModel:
    """
    Initializes a LangChain chat model from a provider configuration
    and wraps it with retry logic if applicable.
    """
    provider = get_provider(model_config.provider)
    model_kwargs = model_config.config or {}
    
    # Get the base model from the provider
    model = provider.get_chat_model(model_config.model_name, **model_kwargs)

    # If it's a Gemini model, chain the .with_retry() method.
    if model_config.provider == "gemini":
        logging.info(f"--- Attaching retry logic to Gemini model: {model_config.model_name} ---")
        
        # Create the runnable with retry logic.
        runnable_with_retries = model.with_retry(
            retry_if_exception_type=(ServerError, DeadlineExceeded),
            wait_exponential_jitter=True, 
            stop_after_attempt=6
        )
        
        # Cast the returned Runnable back to BaseChatModel to satisfy the type checker.
        # This has no runtime effect but fixes the linter error.
        return cast(BaseChatModel, runnable_with_retries)
    
    # For other providers, return the model as is.
    return model

class MockEmbeddings(Embeddings):
    """A mock embeddings class for fast, no-op embedding in debug mode."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 768 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0] * 768
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

async def initialize_llms(params: LLMInitializationParams) -> Tuple:
    """
    Initializes and returns the LLM and embeddings models based on config.
    Returns tuples of (model_instance, model_config) for use with the adapter.
    """
    mock_config = ProviderConfig(provider="mock", model_name="mock")
    app_settings: AppSettings = settings

    if params.coder_debug_mode:
        logging.info("--- 💻 CODER DEBUG MODE ENABLED 💻 ---")
        mock_llm = CoderMockLLM()
        mock_embed = MockEmbeddings()
        return (mock_llm, mock_config), (mock_llm, mock_config), (mock_llm, mock_config), (mock_embed, mock_config)

    if params.debug_mode:
        logging.info("--- 🚀 DEBUG MODE ENABLED 🚀 ---")
        mock_llm = MockLLM()
        mock_embed = MockEmbeddings()
        return (mock_llm, mock_config), (mock_llm, mock_config), (mock_llm, mock_config), (mock_embed, mock_config)

    try:
        logging.info("--- Initializing LLM Providers from config.yaml ---")
        
        main_llm_config = app_settings.llm_providers.main_llm
        llm = _create_chat_model(main_llm_config)
        logging.info(f"--- Main LLM: {main_llm_config.provider} ({main_llm_config.model_name}) ---")
        
        synthesizer_llm_config = app_settings.llm_providers.synthesizer_llm
        synthesizer_llm = _create_chat_model(synthesizer_llm_config)
        logging.info(f"--- Synthesizer LLM: {synthesizer_llm_config.provider} ({synthesizer_llm_config.model_name}) ---")
        
        summarizer_llm_config = app_settings.llm_providers.summarizer_llm
        summarizer_llm = _create_chat_model(summarizer_llm_config)
        logging.info(f"--- Summarizer LLM: {summarizer_llm_config.provider} ({summarizer_llm_config.model_name}) ---")
        
        embeddings_config = app_settings.llm_providers.embeddings_model
        embeddings_provider = get_provider(embeddings_config.provider)
        embeddings_model = embeddings_provider.get_embeddings_model(embeddings_config.model_name)
        logging.info(f"--- Embeddings Model: {embeddings_config.provider} ({embeddings_config.model_name}) ---")

        await llm.ainvoke("Hi")
        logging.info("--- LLM Main Connection Successful Tested ---")

        return (llm, main_llm_config), (synthesizer_llm, synthesizer_llm_config), (summarizer_llm, summarizer_llm_config), (embeddings_model, embeddings_config)

    except Exception as e:
        error_message = f"Failed to initialize LLMs: {e}. Check config.yaml and ensure services are running."
        logging.critical(f"FATAL: {error_message}", exc_info=True)
        raise ConnectionError(error_message)