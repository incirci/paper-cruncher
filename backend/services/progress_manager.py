"""Service for managing real-time progress updates via SSE."""

import asyncio
import json
from typing import List, Dict, Any, AsyncGenerator

class ProgressManager:
    """Manages Server-Sent Events (SSE) for progress tracking."""

    def __init__(self):
        self.listeners: List[asyncio.Queue] = []
        self.active = True

    async def listen(self) -> AsyncGenerator[str, None]:
        """Yields SSE formatted messages to connected clients."""
        q = asyncio.Queue()
        self.listeners.append(q)
        try:
            while self.active:
                msg = await q.get()
                if msg is None:  # Shutdown signal
                    break
                # SSE format: data: <json>\n\n
                yield f"data: {json.dumps(msg)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in self.listeners:
                self.listeners.remove(q)

    async def stop(self):
        """Stop all listeners."""
        self.active = False
        # Iterate over a copy since listeners might remove themselves
        for q in list(self.listeners):
            await q.put(None)
        self.listeners.clear()

    async def emit(self, status: str, progress: float, message: str, task_id: str = "global"):
        """Broadcast a progress update to all listeners.
        
        Args:
            status: 'idle', 'processing', 'completed', 'error'
            progress: 0 to 100
            message: Human readable status message
            task_id: Identifier for the task (e.g. 'indexing', 'uploading')
        """
        msg = {
            "task_id": task_id,
            "status": status,
            "progress": round(progress, 1),
            "message": message
        }
        # Broadcast to all active listeners
        for q in self.listeners:
            await q.put(msg)
