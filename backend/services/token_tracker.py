"""Token tracking and monitoring service."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from backend.models.schemas import TokenUsage, SessionTokenStats


class TokenTracker:
    """Service for tracking and monitoring token usage."""

    def __init__(self, db_path: Path):
        """
        Initialize token tracker.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def reset_all(self) -> None:
        """Delete all token usage records.

        Used by the admin reset endpoint for a clean slate.
        """
        # Ensure schema exists (in case the file was deleted by rmtree)
        self._init_db()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM token_usage")
            conn.commit()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Token usage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    response_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)

            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id
                ON token_usage(session_id)
            """)

            conn.commit()

    def record_usage(self, token_usage: TokenUsage):
        """
        Record token usage for a request.

        Args:
            token_usage: Token usage data to record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO token_usage (session_id, prompt_tokens, response_tokens, total_tokens, model, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    token_usage.session_id,
                    token_usage.prompt_tokens,
                    token_usage.response_tokens,
                    token_usage.total_tokens,
                    token_usage.model,
                    token_usage.timestamp.isoformat(),
                ),
            )

            conn.commit()

    def get_session_stats(self, session_id: str) -> SessionTokenStats:
        """
        Get token statistics for a session.

        Args:
            session_id: Session ID

        Returns:
            Session token statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    COUNT(*) as request_count,
                    SUM(prompt_tokens) as total_prompt,
                    SUM(response_tokens) as total_response,
                    SUM(total_tokens) as total
                FROM token_usage
                WHERE session_id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()

            if row and row[0] > 0:
                return SessionTokenStats(
                    session_id=session_id,
                    request_count=row[0],
                    total_prompt_tokens=row[1] or 0,
                    total_response_tokens=row[2] or 0,
                    total_tokens=row[3] or 0,
                )
            else:
                return SessionTokenStats(session_id=session_id)

    def get_all_usage(self) -> List[TokenUsage]:
        """Get all token usage records."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT session_id, prompt_tokens, response_tokens, total_tokens, model, timestamp
                FROM token_usage
                ORDER BY timestamp DESC
            """)

            usage_records = []
            for row in cursor.fetchall():
                usage_records.append(
                    TokenUsage(
                        session_id=row[0],
                        prompt_tokens=row[1],
                        response_tokens=row[2],
                        total_tokens=row[3],
                        model=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                    )
                )

            return usage_records

    def get_total_usage(self) -> dict:
        """Get total token usage across all sessions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(DISTINCT session_id) as session_count,
                    COUNT(*) as request_count,
                    SUM(prompt_tokens) as total_prompt,
                    SUM(response_tokens) as total_response,
                    SUM(total_tokens) as total
                FROM token_usage
            """)

            row = cursor.fetchone()

            return {
                "session_count": row[0] or 0,
                "request_count": row[1] or 0,
                "total_prompt_tokens": row[2] or 0,
                "total_response_tokens": row[3] or 0,
                "total_tokens": row[4] or 0,
            }

    def check_budget_warning(
        self, session_id: str, budget: int, threshold: float = 0.8
    ) -> tuple[bool, float]:
        """
        Check if session is approaching token budget.

        Args:
            session_id: Session ID
            budget: Token budget limit
            threshold: Warning threshold (0.0 to 1.0)

        Returns:
            Tuple of (is_warning, usage_percentage)
        """
        stats = self.get_session_stats(session_id)
        usage_percentage = stats.total_tokens / budget if budget > 0 else 0.0

        return usage_percentage >= threshold, usage_percentage

    def delete_session_usage(self, session_id: str):
        """Delete all token usage records for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM token_usage WHERE session_id = ?", (session_id,))
            conn.commit()
