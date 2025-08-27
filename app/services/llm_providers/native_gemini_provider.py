# app/services/llm_providers/native_gemini_provider.py

import os
from dotenv import load_dotenv
from google import genai
from .base import LLMProvider
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .native_gemini_wrapper import NativeGeminiChatWrapper

load_dotenv()

class NativeGeminiProvider(LLMProvider):
    """
    Provider that uses the new google-genai SDK, aligning with the latest
    official guidelines and bypassing the legacy google-generativeai library.
    """
    def __init__(self):
        # The new SDK automatically picks up GOOGLE_API_KEY or GEMINI_API_KEY
        # from the environment when genai.Client() is called.
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
        
        # Create a single, persistent client instance for the provider.
        self.client = genai.Client()

    def get_chat_model(self, model_name: str, **kwargs) -> NativeGeminiChatWrapper:
        # Pass the client instance, model name, and any other config to the wrapper.
        return NativeGeminiChatWrapper(
            client=self.client,
            model_name=model_name,
            model_kwargs=kwargs
        )

    def get_embeddings_model(self, model_name: str, **kwargs) -> Embeddings:
        # The langchain-google-genai embeddings component is still appropriate
        # and likely uses the correct, updated APIs under the hood.
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key, **kwargs)