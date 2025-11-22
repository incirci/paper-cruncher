"""Run the FastAPI application."""

import logging

import uvicorn

from backend.core.config import settings


if __name__ == "__main__":
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
    )

