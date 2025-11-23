"""Run the FastAPI application."""

import logging
import signal
import sys

import uvicorn

from backend.core.config import settings

def handle_exit(sig, frame):
    print(f"\nReceived signal {sig}. Exiting...")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers for better Windows Ctrl+C support
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Basic logging configuration so backend services (AIAgent, chat API, etc.)
    # emit INFO-level logs to the console, including our retrieval tracing.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )



    uvicorn.run(
        "backend.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=True,
        timeout_keep_alive=120,  # Increase timeout for streaming
        timeout_graceful_shutdown=3, # Ensure shutdown doesn't hang on open connections
        loop="asyncio", # Explicitly use asyncio loop for better Windows compatibility
    )


