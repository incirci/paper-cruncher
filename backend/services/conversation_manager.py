"""Conversation management service for storing and retrieving chat history."""

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from backend.models.schemas import Conversation, Message, MessageRole


class ConversationManager:
    """Service for managing conversation history and sessions."""

    def __init__(self, db_path: Path):
        """
        Initialize conversation manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Initialize database
        self._init_db()

    def reset_all(self) -> None:
        """Delete all conversations and sessions.

        Used by the admin reset endpoint to ensure there is no
        historical state left in the conversation database.
        """
        # Ensure schema exists (in case the file was deleted by rmtree)
        self._init_db()

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages")
                cursor.execute("DELETE FROM conversations")
                conn.commit()

    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()

                # Conversations table with session_name
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        session_id TEXT PRIMARY KEY,
                        session_name TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        total_tokens INTEGER DEFAULT 0,
                        selected_paper_id TEXT,
                        paper_ids TEXT
                    )
                    """
                )

                # Messages table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        token_count INTEGER,
                        source_papers TEXT,
                        FOREIGN KEY (session_id) REFERENCES conversations (session_id)
                    )
                """)

                conn.commit()

    def create_session(
        self,
        selected_paper_id: Optional[str] = None,
        paper_ids: Optional[list[str]] = None,
        session_name: Optional[str] = None,
    ) -> str:
        """Create a new conversation session.

        Args:
            selected_paper_id: Optional currently selected paper ID
            paper_ids: Optional list of paper IDs associated with this session
            session_name: Optional descriptive name for the session
        """
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        paper_ids_json = json.dumps(paper_ids or [])

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO conversations (session_id, session_name, created_at, updated_at, total_tokens, selected_paper_id, paper_ids)
                    VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                    (session_id, session_name, now, now, selected_paper_id, paper_ids_json),
                )
                conn.commit()

        return session_id

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        token_count: Optional[int] = None,
        source_papers: Optional[List[str]] = None,
    ) -> Message:
        """
        Add a message to a conversation.

        Args:
            session_id: Session ID
            role: Message role (user/assistant)
            content: Message content
            token_count: Optional token count for this message
            source_papers: Optional list of source papers referenced

        Returns:
            Created message
        """
        timestamp = datetime.now()
        source_papers_json = json.dumps(source_papers or [])

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()

                # Insert message
                cursor.execute(
                    """
                    INSERT INTO messages (session_id, role, content, timestamp, token_count, source_papers)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        session_id,
                        role.value,
                        content,
                        timestamp.isoformat(),
                        token_count,
                        source_papers_json,
                    ),
                )

                # Update conversation
                cursor.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?,
                        total_tokens = total_tokens + ?
                    WHERE session_id = ?
                """,
                    (timestamp.isoformat(), token_count or 0, session_id),
                )

                conn.commit()

        return Message(
            role=role,
            content=content,
            timestamp=timestamp,
            token_count=token_count,
            source_papers=source_papers or [],
        )

    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """
        Get a conversation with all its messages.

        Args:
            session_id: Session ID

        Returns:
            Conversation object or None if not found
        """
        # Reads are generally safe without lock in WAL mode or with short writes,
        # but using the lock ensures we don't read while a write is partially happening (though SQLite transactions handle that).
        # To be safe and simple, we can lock reads too, or just rely on SQLite.
        # Given the low volume, locking reads is fine and guarantees no "database locked" on connect.
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()

                # Get conversation
                cursor.execute(
                    """
                    SELECT session_id, session_name, created_at, updated_at, total_tokens, selected_paper_id, paper_ids
                    FROM conversations
                    WHERE session_id = ?
                """,
                    (session_id,),
                )

                conv_row = cursor.fetchone()
                if not conv_row:
                    return None

                # Get messages
                cursor.execute(
                    """
                    SELECT id, role, content, timestamp, token_count, source_papers
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """,
                    (session_id,),
                )

                messages = []
                for row in cursor.fetchall():
                    messages.append(
                        Message(
                            id=row[0],
                            role=MessageRole(row[1]),
                            content=row[2],
                            timestamp=datetime.fromisoformat(row[3]),
                            token_count=row[4],
                            source_papers=json.loads(row[5]),
                        )
                    )

                paper_ids: list[str] = []
                if conv_row[6]:
                    try:
                        paper_ids = json.loads(conv_row[6])
                    except Exception:
                        paper_ids = []

                return Conversation(
                    session_id=conv_row[0],
                    session_name=conv_row[1],
                    created_at=datetime.fromisoformat(conv_row[2]),
                    updated_at=datetime.fromisoformat(conv_row[3]),
                    total_tokens=conv_row[4],
                    messages=messages,
                    selected_paper_id=conv_row[5],
                    paper_ids=paper_ids,
                )

    def get_conversation_history(self, session_id: str) -> List[Message]:
        """Get conversation history (messages only)."""
        conversation = self.get_conversation(session_id)
        return conversation.messages if conversation else []

    def delete_messages(self, session_id: str) -> None:
        """Delete all messages for a session but keep the session row.

        Also resets the accumulated token count and updates the session's
        updated_at timestamp. This is used when the paper set changes but
        we want to keep the session metadata intact.
        """
        now = datetime.now().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                cursor.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?, total_tokens = 0
                    WHERE session_id = ?
                    """,
                    (now, session_id),
                )
                conn.commit()

    def delete_conversation(self, session_id: str):
        """Delete a conversation and all its messages."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()

                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                cursor.execute(
                    "DELETE FROM conversations WHERE session_id = ?", (session_id,)
                )

                conn.commit()

    def list_sessions(self) -> List[dict]:
        """List all conversation sessions with message counts."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT c.session_id, c.session_name, c.created_at, c.updated_at, c.total_tokens, 
                           c.selected_paper_id, c.paper_ids,
                           COUNT(m.id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.session_id = m.session_id
                    GROUP BY c.session_id
                    ORDER BY c.updated_at DESC
                """)

                sessions = []
                for row in cursor.fetchall():
                    sessions.append(
                        {
                            "session_id": row[0],
                            "session_name": row[1],
                            "created_at": row[2],
                            "updated_at": row[3],
                            "total_tokens": row[4],
                            "selected_paper_id": row[5],
                            "paper_ids": row[6],
                            "message_count": row[7],
                        }
                    )

                return sessions

    def update_session_papers(
        self,
        session_id: str,
        selected_paper_id: Optional[str],
        paper_ids: list[str],
    ) -> None:
        """Update paper context for an existing session."""
        paper_ids_json = json.dumps(paper_ids or [])
        now = datetime.now().isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE conversations
                    SET selected_paper_id = ?,
                        paper_ids = ?,
                        updated_at = ?
                    WHERE session_id = ?
                    """,
                    (selected_paper_id, paper_ids_json, now, session_id),
                )
                conn.commit()

    def rename_session(self, session_id: str, session_name: str) -> None:
        """Rename a session.
        
        Args:
            session_id: Session ID to rename
            session_name: New name for the session
        """
        now = datetime.now().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE conversations
                    SET session_name = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (session_name, now, session_id),
                )
                conn.commit()

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM conversations WHERE session_id = ?", (session_id,)
                )
                return cursor.fetchone() is not None

    def duplicate_session(self, source_session_id: str) -> str:
        """Duplicate a session (papers only, no messages).
        
        Args:
            source_session_id: ID of the session to duplicate
            
        Returns:
            New session ID
        """
        source_conv = self.get_conversation(source_session_id)
        if not source_conv:
            raise ValueError("Source session not found")
            
        new_name = f"Copy of {source_conv.session_name}" if source_conv.session_name else "Copy of Session"
        
        return self.create_session(
            selected_paper_id=source_conv.selected_paper_id,
            paper_ids=source_conv.paper_ids,
            session_name=new_name
        )

    def delete_message(self, message_id: int) -> bool:
        """Delete a specific message by ID.
        
        Returns:
            True if message was deleted, False if not found.
        """
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
                conn.commit()
                return cursor.rowcount > 0
