"""Conversation management service for storing and retrieving chat history."""

import json
import sqlite3
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

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_tokens INTEGER DEFAULT 0
                )
            """)

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

    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversations (session_id, created_at, updated_at, total_tokens)
                VALUES (?, ?, ?, 0)
            """,
                (session_id, now, now),
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

        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get conversation
            cursor.execute(
                """
                SELECT session_id, created_at, updated_at, total_tokens
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
                SELECT role, content, timestamp, token_count, source_papers
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
                        role=MessageRole(row[0]),
                        content=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        token_count=row[3],
                        source_papers=json.loads(row[4]),
                    )
                )

            return Conversation(
                session_id=conv_row[0],
                created_at=datetime.fromisoformat(conv_row[1]),
                updated_at=datetime.fromisoformat(conv_row[2]),
                total_tokens=conv_row[3],
                messages=messages,
            )

    def get_conversation_history(self, session_id: str) -> List[Message]:
        """Get conversation history (messages only)."""
        conversation = self.get_conversation(session_id)
        return conversation.messages if conversation else []

    def delete_conversation(self, session_id: str):
        """Delete a conversation and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute(
                "DELETE FROM conversations WHERE session_id = ?", (session_id,)
            )

            conn.commit()

    def list_sessions(self) -> List[dict]:
        """List all conversation sessions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT session_id, created_at, updated_at, total_tokens
                FROM conversations
                ORDER BY updated_at DESC
            """)

            sessions = []
            for row in cursor.fetchall():
                sessions.append(
                    {
                        "session_id": row[0],
                        "created_at": row[1],
                        "updated_at": row[2],
                        "total_tokens": row[3],
                    }
                )

            return sessions

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM conversations WHERE session_id = ?", (session_id,)
            )
            return cursor.fetchone() is not None
