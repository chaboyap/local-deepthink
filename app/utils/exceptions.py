# app/utils/exceptions.py

class AgentExecutionError(Exception):
    """Custom exception raised when a critical node fails all retry attempts."""
    def __init__(self, message: str, node_id: str):
        self.message = message
        self.node_id = node_id
        super().__init__(f"Critical failure in node '{node_id}': {message}")