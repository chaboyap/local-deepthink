# app/core/ui_logging.py
import json
import logging
from typing import Dict, Any

class UIJSONFormatter(logging.Formatter):
    """
    A custom JSON formatter for sending structured log messages to the UI.

    This formatter is the core of the refactored logging system. It ensures
    that the UI stream is not just a simple text log, but a command and
    control channel for UI state updates.

    How it works:
    1. It overrides the default `format` method.
    2. It checks if a `LogRecord` has a special `ui_extra` attribute.
       This attribute is populated by using the `extra` kwarg in logging calls,
       e.g., `logging.info("...", extra={'ui_extra': {'type': 'graph', 'data': ...}})`
    3. If `ui_extra` exists and is a dictionary, it's used directly as the
       JSON payload. This allows for sending structured commands like 'graph',
       'perplexity', or 'session_id'.
    4. If `ui_extra` is not found, it defaults to a standard log message
       format: `{"type": "log", "data": "log message here"}`.
    5. The final dictionary is serialized into a JSON string, which is what
       gets put onto the asyncio.Queue for the UI to consume.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record into a JSON string.

        Args:
            record: The LogRecord instance to format.

        Returns:
            A JSON-formatted string representing the log message or UI command.
        """
        payload: Dict[str, Any]
        # Check for our special 'ui_extra' dictionary in the record.
        ui_extra_data = getattr(record, 'ui_extra', None)

        if isinstance(ui_extra_data, dict):
            # If it's a dict, we use it directly. This is our command channel.
            payload = ui_extra_data
        else:
            # Otherwise, create a standard log payload.
            payload = {
                "type": "log",
                "data": record.getMessage()
            }

        return json.dumps(payload)