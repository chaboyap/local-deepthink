# main.py

import uvicorn
import yaml
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import router as api_router
from fastapi.middleware.cors import CORSMiddleware
from app.core.state_manager import session_manager
from app.core.logging_handler import AsyncQueueHandler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse 
from app.core.ui_logging import UIJSONFormatter

# --- CONFIGURE LOGGING ---

# Get the root logger. All loggers in the application will inherit from this.
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set the minimum level of messages to capture

# 1. CONFIGURE FILE HANDLER (for plain text logs)
# This handler writes verbose, plain text logs to a rotating file.
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler = RotatingFileHandler(
    'app_activity.log', 
    maxBytes=10*1024*1024*5, # 50 Megabytes
    backupCount=5, 
    encoding='utf-8'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler) # Add to the root logger

# 2. CONFIGURE QUEUE HANDLER (for structured UI messages)
# This handler uses our custom UIJSONFormatter to create JSON payloads
# and sends them to the UI's asyncio.Queue. This is the key to preserving
# UI command and control functionality (graphs, metrics, etc.).
ui_json_formatter = UIJSONFormatter()
queue_handler = AsyncQueueHandler(session_manager.log_stream)
queue_handler.setFormatter(ui_json_formatter)
logger.addHandler(queue_handler) # Add to the root logger

#google_logger = logging.getLogger('google.genai')
# Set its level to WARNING to suppress informational messages
#google_logger.setLevel(logging.WARNING)

# --- END OF LOGGING CONFIGURATION ---

# Create the FastAPI application instance
app = FastAPI(title="Network of Agents (NoA)")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Catches and logs Pydantic validation errors before they reach the user.
    This ensures that bad requests are logged in our application log file.
    """
    # Log the detailed error information to the file log. The UI will get a 422 response.
    logger.error(f"API Validation Error: {exc.errors()}", extra={"ui_extra": None})
    
    # Return the default 422 response that FastAPI would have sent anyway
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# --- CORS MIDDLEWARE CONFIGURATION ---
# It allows your browser's JavaScript to communicate with the Python backend.
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- END OF CORS SECTION ---

# Mount the static directory to serve files like index.html, css, js
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include the API router from the app.api.endpoints module
app.include_router(api_router)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """
    Serves the main HTML file for the user interface.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

if __name__ == "__main__":
    # This block allows running the app directly with `python main.py`
    # It's useful for development but `uvicorn main:app --reload` is preferred
    uvicorn.run(app, host="0.0.0.0", port=8000)