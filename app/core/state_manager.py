# app/core/state_manager.py

import asyncio
import uuid
import logging
import redis.asyncio as redis
from typing import Dict, Any, Optional
from cachetools import TTLCache, LRUCache
from collections import defaultdict
from app.rag.raptor import RAPTOR

from app.core.config import settings
from app.core.context import ServiceContext
from app.services.llm_service import initialize_llms, LLMInitializationParams
from app.core.serialization import serialize_state, deserialize_state

class SessionManager:
    """
    [ROLE]: Central authority for session state, live services, and concurrency.

    [ARCHITECTURAL_PATTERN]: State Hydrator & Non-Serializable Boundary Controller.
    This class manages the boundary between the serializable `GraphState` (for persistence)
    and the live `ServiceContext` (for runtime operations). It injects services
    into the state object only when it is active in memory.

    [CORE_RESPONSIBILITIES]:
      - [STATE_LIFECYCLE]: Abstract persistence of session state via Redis or in-memory dict.
          - Method: `create_session`, `update_session_state`, `set_final_state`
      - [SERVICE_HYDRATION]: Re-initialize and inject `ServiceContext` on session load.
          - Method: `get_session`
      - [RESILIENCE]: Rebuild live RAPTOR RAG index from persisted documents on cache miss (e.g., post-restart).
          - Method: `get_session` (internal logic)
      - [RESOURCE_MANAGEMENT]: Cache live RAG indexes in a memory-bounded LRUCache to prevent OOM errors.
          - Method: `store_rag_index`
      - [CONCURRENCY_CONTROL]: Vend session-specific `asyncio.Lock`s to prevent race conditions.
          - Attribute: `session_locks`
      - [UI_STREAMING]: Own the `asyncio.Queue` for routing structured logs to the frontend.
          - Attribute: `log_stream`

    [PUBLIC_API]:
      - `create_session(initial_state: dict) -> str`: Persists initial state, returns new session_id.
      - `get_session(session_id: str) -> dict | None`: Retrieves state from persistence and hydrates it with live services.
      - `update_session_state(session_id: str, node_output: dict)`: Atomically updates persisted state.
      - `set_final_state(session_id: str, final_state: dict)`: Overwrites state with final graph output.
      - `store_rag_index(session_id: str, raptor_index: RAPTOR)`: Caches a live RAG index.
      - `session_locks[session_id]`: Accessor for a session's `asyncio.Lock`.
    """
    def __init__(self):
        if settings.persistence.enabled:
            self.redis_client = redis.from_url(settings.persistence.redis_url)
            logging.info(f"--- Persistence ENABLED. Connecting to Redis at {settings.persistence.redis_url} ---")
        else:
            self.sessions_in_memory: Dict[str, Dict[str, Any]] = {}
            logging.warning("--- Persistence DISABLED. Sessions will be lost on application restart. ---")
        
        self.log_stream = asyncio.Queue()
        self.final_reports: TTLCache[str, Dict[str, str]] = TTLCache(maxsize=1024, ttl=settings.hyperparameters.session_ttl_seconds)
        # NEW: Caches for live, non-serializable objects
        self.live_rag_indexes: LRUCache[str, RAPTOR] = LRUCache(maxsize=100)
        self.session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def create_session(self, initial_state: Dict[str, Any]) -> str:
        """Creates a new session by storing its state in Redis or memory."""
        session_id = str(uuid.uuid4())
        initial_state["session_id"] = session_id

        state_json = serialize_state(initial_state)
        
        if settings.persistence.enabled:
            await self.redis_client.set(
                f"session:{session_id}", 
                state_json, 
                ex=settings.hyperparameters.session_ttl_seconds
            )
        else:
            self.sessions_in_memory[session_id] = {"state_json": state_json}

        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        [ARCHITECTURAL_ROLE]: State Hydrator
        [PRIMARY_TASK]: Retrieve serialized state from persistence and inject live,
                    non-serializable services to make it runtime-ready.
        [RESILIENCE_MECHANISM]: On RAG index cache miss (e.g., post-restart),
                                this method rebuilds the live index from
                                persisted documents (`all_rag_documents`).
        """
        state_json = None
        if settings.persistence.enabled:
            state_json_bytes = await self.redis_client.get(f"session:{session_id}")
            if state_json_bytes:
                state_json = state_json_bytes.decode('utf-8')
        else:
            session_data = self.sessions_in_memory.get(session_id)
            if session_data:
                state_json = session_data["state_json"]

        if not state_json:
            return None

        state = deserialize_state(state_json)
        params = state.get("params")
        if not params:
            raise ValueError("Session state is corrupt: missing 'params'.")
            
        init_params = LLMInitializationParams(debug_mode=params.debug_mode, coder_debug_mode=params.coder_debug_mode)
        (llm, _), (synth_llm, _), (summ_llm, _), (embed_model, _) = await initialize_llms(init_params)
        
        service_context = ServiceContext(
            llm=llm, 
            synthesizer_llm=synth_llm, 
            summarizer_llm=summ_llm, 
            embeddings_model=embed_model
        )
        
        # Check for a live RAG index in the cache; if not found, rebuild it.
        if session_id in self.live_rag_indexes:
            service_context.raptor_index = self.live_rag_indexes[session_id]
        elif state.get("all_rag_documents"):
            # This logic block makes the RAG index resilient to server restarts.
            logging.info(f"RAG index for session {session_id} not in live cache. Rebuilding...")
            rebuilt_index = RAPTOR(llm=service_context.summarizer_llm, embeddings_model=service_context.embeddings_model)
            await rebuilt_index.add_documents(state["all_rag_documents"])
            self.live_rag_indexes[session_id] = rebuilt_index # Store in cache
            service_context.raptor_index = rebuilt_index

        state['services'] = service_context
        return state

    async def update_session_state(self, session_id: str, node_output: Dict[str, Any]):
        """
        Atomically updates the state of a specific session in Redis or memory.
        """
        if settings.persistence.enabled:
            async with self.redis_client.pipeline(transaction=True) as pipe:
                session_key = f"session:{session_id}"
                await pipe.watch(session_key)
                state_json_bytes = await pipe.get(session_key)
                if not state_json_bytes:
                    return 

                state = deserialize_state(state_json_bytes.decode('utf-8'))
                self._apply_updates_to_state(state, node_output)
                updated_state_json = serialize_state(state)
                
                pipe.multi()
                await pipe.set(session_key, updated_state_json, ex=settings.hyperparameters.session_ttl_seconds)
                try:
                    await pipe.execute()
                except redis.exceptions.WatchError:
                    logging.warning(f"WatchError on session {session_id}, update failed. Another process may have updated it.")

        else: # In-memory update
            session_data = self.sessions_in_memory.get(session_id)
            if not session_data:
                return
            state = deserialize_state(session_data["state_json"])
            self._apply_updates_to_state(state, node_output)
            session_data["state_json"] = serialize_state(state)
    
    def _apply_updates_to_state(self, state: Dict[str, Any], node_output: Dict[str, Any]):
        """Internal helper to apply node output to a state dictionary."""
        for key, value in node_output.items():
            if key in ['agent_outputs', 'memory', 'critiques'] and isinstance(state.get(key), dict):
                state[key].update(value)
            elif key == 'all_rag_documents' and isinstance(value, list):
                if state.get(key) is None: state[key] = []
                state[key].extend(value)
            else:
                state[key] = value

    async def set_final_state(self, session_id: str, final_state: Dict[str, Any]):
        """Sets the final state of a session after the graph run completes."""
        final_state_json = serialize_state(final_state)
        if settings.persistence.enabled:
            await self.redis_client.set(
                f"session:{session_id}", 
                final_state_json, 
                ex=settings.hyperparameters.session_ttl_seconds
            )
        else:
            if session_id in self.sessions_in_memory:
                 self.sessions_in_memory[session_id]["state_json"] = final_state_json

    def store_rag_index(self, session_id: str, raptor_index: RAPTOR):
        """Stores a newly built RAPTOR index in the live cache."""
        self.live_rag_indexes[session_id] = raptor_index

    def store_report(self, session_id: str, papers: Dict[str, str]):
        """Stores the final generated report for a session."""
        self.final_reports[session_id] = papers

    def get_report(self, session_id: str) -> Dict[str, str] | None:
        """Retrieves a final report by session ID."""
        return self.final_reports.get(session_id)

session_manager = SessionManager()