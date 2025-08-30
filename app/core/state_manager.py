# app/core/state_manager.py

import asyncio
import uuid
import logging
import random
from redis import asyncio as aredis, WatchError 
from typing import Dict, Any, Optional
from cachetools import TTLCache, LRUCache
from collections import defaultdict
from app.rag.raptor import RAPTOR

from app.core.config import settings
from app.core.context import ServiceContext
from app.services.llm_service import initialize_llms, LLMInitializationParams
from app.core.serialization import serialize_state, deserialize_state

class SessionManager:
    """Manages the lifecycle, persistence, and live context of user sessions.

    This class serves as the central authority for all session-related state. It
    handles the critical boundary between the serializable GraphState, which is
    persisted to Redis, and the live, non-serializable ServiceContext (e.g.,
    LLM clients, RAG indexes), which is injected at runtime.

    This "hydration" pattern makes sessions resilient to application restarts,
    as live services can be rebuilt from the persisted state. It also manages
    concurrency for state updates and provides the central queue for streaming
    UI logs.
    """
    def __init__(self):
        if settings.persistence.enabled:
            self.redis_client = aredis.from_url(settings.persistence.redis_url)
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
        """Retrieves and "hydrates" a session state, making it runtime-ready.
        This method retrieves the serialized state, deserializes it, initializes
        the live ServiceContext (LLMs, etc.), rebuilds the RAG index if needed,
        and injects the context into the state object under the 'services' key.
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

    _UPDATE_RETRIES = 3
    _RETRY_BACKOFF_FACTOR = 0.1

    async def _attempt_atomic_update(self, session_id: str, node_output: Dict[str, Any]) -> bool:
        """Attempts a single, atomic read-modify-write operation in Redis.
        This internal helper implements the core Redis transaction logic using WATCH.
        """
        async with self.redis_client.pipeline(transaction=True) as pipe:
            session_key = f"session:{session_id}"
            try:
                # Set up the watch
                await pipe.watch(session_key)

                # The "read" part of the cycle
                state_json_bytes = await pipe.get(session_key)
                if not state_json_bytes:
                    logging.warning(f"Session {session_id} expired or was deleted during update attempt.")
                    return True # Nothing to update, so it's a "success"

                state = deserialize_state(state_json_bytes.decode('utf-8'))

                # The "modify" part of the cycle
                self._apply_updates_to_state(state, node_output)
                updated_state_json = serialize_state(state)

                # The "write" part of the cycle (staged in the transaction)
                pipe.multi()
                pipe.set(session_key, updated_state_json, ex=settings.hyperparameters.session_ttl_seconds)

                # Attempt to execute the transaction
                await pipe.execute()
                return True # Success!

            except WatchError:
                # The key was modified by another client after we WATCHed it.
                # The transaction was aborted.
                return False # Indicate failure due to collision.

    async def update_session_state(self, session_id: str, node_output: Dict[str, Any]):
        """Atomically updates a session's state, retrying on data race collisions.
        This method merges node output into the persisted session state. It uses an
        optimistic locking pattern with exponential backoff to handle concurrent
        writes gracefully.
        """
        if settings.persistence.enabled:
            for attempt in range(self._UPDATE_RETRIES):
                success = await self._attempt_atomic_update(session_id, node_output)
                if success:
                    # Log success only on the first attempt for cleaner logs
                    if attempt == 0:
                        logging.debug(f"State for session {session_id} updated successfully on first try.")
                    else:
                        logging.info(f"State for session {session_id} updated successfully after {attempt + 1} attempts.")
                    return # Exit successfully

                # If we failed, wait before retrying
                logging.warning(f"Collision detected for session {session_id} (Attempt {attempt + 1}/{self._UPDATE_RETRIES}). Retrying...")
                # [KISS]: Simple exponential backoff with jitter to prevent thundering herd
                backoff_time = self._RETRY_BACKOFF_FACTOR * (2 ** attempt)
                jitter = random.uniform(0, backoff_time * 0.1)
                await asyncio.sleep(backoff_time + jitter)

            logging.error(f"FATAL: Failed to update state for session {session_id} after {self._UPDATE_RETRIES} attempts. Update was dropped.")

        else: # In-memory update remains the same (no concurrency issue to solve)
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