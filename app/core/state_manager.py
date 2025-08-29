# app/core/state_manager.py

import asyncio
import uuid
from typing import Dict, Any
from cachetools import TTLCache
from app.core.config import settings
from app.core.context import ServiceContext

class SessionManager:
    """
    Manages the state of all active graph execution sessions.
    This version is hardened against race conditions and memory leaks.
    
    NOTE: This class has been refactored. The `log` method was removed to centralize
    all logging through Python's standard `logging` module. The `log_stream` queue
    is now populated by a dedicated `AsyncQueueHandler` in main.py.
    """
    def __init__(self):
        # Sessions will now automatically expire after a configured time (e.g., 2 days)
        session_ttl_seconds = settings.hyperparameters.session_ttl_seconds
        self.sessions: TTLCache[str, Dict[str, Any]] = TTLCache(maxsize=1024, ttl=session_ttl_seconds)
        self.log_stream = asyncio.Queue()
        # The final reports should also expire to prevent a memory leak.
        self.final_reports: TTLCache[str, Dict[str, str]] = TTLCache(maxsize=1024, ttl=session_ttl_seconds)

    def create_session(self, initial_state: Dict[str, Any], service_context: ServiceContext) -> str:
        """Creates a new session with separated state and services."""
        session_id = str(uuid.uuid4())
        initial_state["session_id"] = session_id

        # This lock is unique to each session and will be used by the API endpoints
        self.sessions[session_id] = {
            "state": initial_state,
            "services": service_context,
            "lock": asyncio.Lock()
        }
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any] | None:
        """
        Retrieves the entire session object (state and lock) by its ID.
        The caller is now responsible for accessing session['state'].
        """
        return self.sessions.get(session_id)

    async def update_session_state(self, session_id: str, node_output: Dict[str, Any]):
        """
        Atomically updates the state of a specific session.
        This method is now thread-safe by acquiring the session's lock.
        """
        session = self.get_session(session_id)
        if not session:
            return

        async with session["lock"]:
            # Operate on the nested state dictionary
            state = session["state"]
            for key, value in node_output.items():
                if key in ['agent_outputs', 'memory', 'critiques'] and isinstance(state.get(key), dict):
                    # Use update for appending to dictionaries
                    state[key].update(value)
                elif key == 'all_rag_documents' and isinstance(value, list):
                    # Ensure we are extending lists, not replacing them
                    if state.get(key) is None or not isinstance(state.get(key), list):
                        state[key] = []
                    state[key].extend(value)
                else:
                    state[key] = value

    def set_final_state(self, session_id: str, final_state: Dict[str, Any]):
        """Sets the final state of a session after the graph run completes."""
        session = self.get_session(session_id)
        if session:
            # Only update the state, leave the lock and services intact
            session["state"] = final_state

    def store_report(self, session_id: str, papers: Dict[str, str]):
        """Stores the final generated report for a session."""
        self.final_reports[session_id] = papers

    def get_report(self, session_id: str) -> Dict[str, str] | None:
        """Retrieves a final report by session ID."""
        return self.final_reports.get(session_id)

# Global instance
session_manager = SessionManager()