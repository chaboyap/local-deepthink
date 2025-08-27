# app/core/config.py

import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, Optional, List

# --- Pydantic Models for Type-Safe Configuration ---

class ProviderConfig(BaseModel):
    provider: str
    model_name: str
    enable_structured_output: Optional[bool] = False
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class LLMProviders(BaseModel):
    main_llm: ProviderConfig
    synthesizer_llm: ProviderConfig
    summarizer_llm: ProviderConfig
    embeddings_model: ProviderConfig

class Debugging(BaseModel):
    debug_mode: bool = False
    coder_debug_mode: bool = False

class Timeouts(BaseModel):
    agent_timeout_seconds: float = Field(..., gt=0)
    reflection_timeout_seconds: float = Field(..., gt=0)
    synthesis_timeout_seconds: float = Field(..., gt=0)

class Hyperparameters(BaseModel):
    cot_trace_depth: int
    num_epochs: int
    num_questions_for_harvest: int
    vector_word_size: int
    prompt_alignment: float
    density: float
    critique_strategy: str
    learning_rate: float
    timeouts: Timeouts
    default_prompt: Optional[str] = None
    default_mbti_selection: Optional[List[str]] = None
    session_ttl_seconds: int = 172800 # Some runs may be 2 days long.

class AppSettings(BaseModel):
    """The main configuration model that holds all settings."""
    llm_providers: LLMProviders
    debugging: Debugging
    hyperparameters: Hyperparameters

# --- Singleton Loader ---

class Config:
    _instance = None
    _settings: AppSettings

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Loads and validates the configuration from YAML."""
        try:
            with open('config.yaml', 'r') as f:
                raw_config = yaml.safe_load(f)
            # Pydantic does the validation and type casting here
            self._settings = AppSettings(**raw_config)
        except FileNotFoundError:
            raise RuntimeError("CRITICAL: Configuration file 'config.yaml' not found.")
        except ValidationError as e:
            # This is the "fail-fast" mechanism. The app will not start.
            raise RuntimeError(f"CRITICAL: Configuration error in 'config.yaml':\n{e}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: An unexpected error occurred while loading config: {e}")
            
    def get(self) -> AppSettings:
        """Returns the fully validated settings object."""
        return self._settings

# Global, validated instance. Import this elsewhere.
settings = Config().get()