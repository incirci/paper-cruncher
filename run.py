"""Run the FastAPI application."""

import uvicorn

from backend.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=True,
        timeout_keep_alive=120,  # Increase timeout for streaming
    )
