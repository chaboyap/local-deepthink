# app/core/logging_handler.py

import logging
import asyncio
from typing import Optional

class AsyncQueueHandler(logging.Handler):
    """
    A robust, thread-safe logging handler that puts records into an asyncio.Queue.

    This handler is designed to bridge the gap between standard synchronous logging
    calls (which can occur in any thread) and an asyncio event loop. It uses
    `loop.call_soon_threadsafe()` to safely schedule the log message to be
    added to the queue on the main event loop, preventing cross-thread concurrency
    issues.
    """
    def __init__(self, queue: asyncio.Queue):
        """
        Initializes the handler.

        It captures the running asyncio event loop at the time of its creation,
        which is expected to be the main application loop.

        Args:
            queue: The asyncio.Queue to which log records will be sent.
        """
        super().__init__()
        self.queue = queue
        try:
            # Get the running event loop. This should be the main loop
            # where the application is running.
            self._loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            # This can happen if the handler is initialized from a thread
            # with no running loop. We'll fall back to getting the main loop.
            self._loop = asyncio.get_event_loop_policy().get_event_loop()


    def emit(self, record: logging.LogRecord) -> None:
        """
        Formats the log record and safely schedules it to be put into the queue.

        This method is thread-safe. It handles potential `QueueFull` exceptions
        to prevent the logger from crashing.

        Args:
            record: The log record to be processed.
        """
        if self._loop is None:
            # Failsafe in case no loop could be found.
            print(f"ERROR: No asyncio loop found for AsyncQueueHandler. Log message dropped: {self.format(record)}")
            return

        def enqueue_record():
            """Wrapper to handle potential QueueFull errors."""
            try:
                self.queue.put_nowait(self.format(record))
            except asyncio.QueueFull:
                # This is a critical failsafe. If the queue is full
                # (e.g., UI is disconnected or slow), we should not crash.
                # We log the drop to stderr as a last resort.
                print(f"WARNING: Log queue is full. Log message dropped: {self.format(record)}")

        # This is the key to thread safety. It schedules the `enqueue_record`
        # coroutine to be run on the event loop this handler was created in,
        # regardless of which thread `emit` was called from.
        self._loop.call_soon_threadsafe(enqueue_record)