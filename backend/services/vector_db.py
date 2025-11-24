"""Vector database service for semantic search using ChromaDB."""

import threading
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.models.schemas import PaperChunk, PaperMetadata


class VectorDBService:
    """Service for managing vector database operations."""

    def __init__(self, db_path: Path):
        """
        Initialize ChromaDB client.

        Args:
            db_path: Path to ChromaDB storage directory
        """
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Create or get collection for paper chunks
        self.collection = self.client.get_or_create_collection(
            name="paper_chunks",
            metadata={"description": "Chunks from journal articles"},
        )

    def add_paper_chunks(self, chunks: List[PaperChunk], paper_metadata: PaperMetadata):
        """
        Add paper chunks to the vector database.

        Args:
            chunks: List of paper chunks to add
            paper_metadata: Metadata about the paper
        """
        if not chunks:
            return

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "paper_id": chunk.paper_id,
                "paper_filename": paper_metadata.filename,
                "paper_title": getattr(paper_metadata, "canonical_title", paper_metadata.filename),
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number or 0,
            }
            for chunk in chunks
        ]

        # Add to collection (ChromaDB handles embedding automatically)
        with self._lock:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(
        self,
        query: str,
        n_results: int = 5,
        paper_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Search for relevant chunks using semantic search.

        Args:
            query: Search query
            n_results: Number of results to return
            paper_ids: Optional list of paper IDs to filter by

        Returns:
            List of search results with content and metadata
        """
        # Build where filter if paper_ids provided
        where_filter = None
        if paper_ids:
            where_filter = {"paper_id": {"$in": paper_ids}}

        # Perform search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )

        return formatted_results

    def delete_paper(self, paper_id: str):
        """
        Delete all chunks for a specific paper.

        Args:
            paper_id: ID of the paper to delete
        """
        # Get all chunk IDs for this paper
        # Note: .get() is a read op, but we want to ensure we don't delete while someone else is reading/writing if possible,
        # though Chroma handles readers. The critical part is the delete.
        # We'll lock the whole operation to be safe and consistent.
        with self._lock:
            results = self.collection.get(where={"paper_id": paper_id})

            if results["ids"]:
                self.collection.delete(ids=results["ids"])

    def get_paper_chunks(self, paper_id: str) -> List[dict]:
        """
        Get all chunks for a specific paper.

        Args:
            paper_id: ID of the paper

        Returns:
            List of chunks with content and metadata
        """
        results = self.collection.get(where={"paper_id": paper_id})

        chunks = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                chunks.append(
                    {
                        "id": results["ids"][i],
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )

        return chunks

    def get_all_papers(self) -> List[dict]:
        """
        Get list of all unique papers in the database.

        Returns:
            List of dictionaries with paper_id, paper_filename and paper_title
        """
        # Get all metadata (limit to 10k to avoid truncation, though unlikely to hit soon)
        # ChromaDB's default limit can vary, so we explicit set a high one.
        results = self.collection.get(limit=10000, include=["metadatas"])
        
        # Extract unique papers
        papers = {}
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                paper_id = metadata.get("paper_id")
                paper_filename = metadata.get("paper_filename")
                paper_title = metadata.get("paper_title") or paper_filename
                if paper_id and paper_filename:
                    papers[paper_id] = {
                        "paper_filename": paper_filename,
                        "paper_title": paper_title,
                    }

        return [
            {
                "paper_id": pid,
                "paper_filename": data["paper_filename"],
                "paper_title": data["paper_title"],
            }
            for pid, data in papers.items()
        ]

    def get_paper_summaries(self, paper_ids: Optional[List[str]] = None) -> List[dict]:
        """Get lightweight, concept-focused summaries of papers.

        Args:
            paper_ids: Optional list of paper IDs to summarize. If None, summarizes all.

        Returns:
            List of dictionaries with paper_id, filename, canonical title,
            and summary content.
        """
        papers = self.get_all_papers()
        
        # Filter papers if specific IDs requested
        if paper_ids is not None:
            target_set = set(paper_ids)
            papers = [p for p in papers if p["paper_id"] in target_set]

        summaries: List[dict] = []

        # Simple, domain-agnostic keywords that often signal substantive content
        # and concept-dense descriptions (phenomena, variables, tasks, datasets,
        # equipment, methods, conclusions). Keep this lightweight and heuristic.
        strong_keywords = [
            # Problem / phenomenon / task
            "we investigate",
            "we study",
            "we analyze",
            "we examine",
            "our objective",
            "the goal of this",
            "the aim of this",

            # Methods / models / methodology
            "we propose",
            "we present",
            "we develop",
            "we introduce",
            "our method",
            "our approach",
            "our model",
            "our framework",
            "methodology",
            "experimental setup",
            "protocol",
            "pipeline",

            # Data / variables / equipment (domain-agnostic)
            "dataset",
            "variables",
            "indicator",
            "test",
            "measurements",
            "signals",
            "recordings",
            "method", 
            "methodology",
            "apparatus",
            "equipment",

            # Results / conclusions
            "we show",
            "we demonstrate",
            "we find",
            "our results",
            "results show",
            "our findings",
            "the results indicate",
            "we conclude",
            "we conclude that",
            "in conclusion",
            "these findings",

            # Generic but still contribution-focused phrases
            "this work",
            "this study",
            "in this study",
            "in this paper we",
        ]

        def score_chunk(text: str, index: int, total: int) -> float:
            """Heuristically score a chunk for micro-summary selection."""
            if not text:
                return 0.0

            t = text.lower()
            score = 0.0

            # Prefer chunks with strong keywords (indicative of contributions/results)
            for kw in strong_keywords:
                if kw in t:
                    score += 3.0

            # Light preference for chunks with multiple sentences (richer content)
            sentence_separators = t.count(".") + t.count("!") + t.count("?")
            if sentence_separators > 1:
                score += 1.0

            # Positional bias: early, middle, and late chunks
            if index == 0:
                score += 1.5
            elif index == total - 1:
                score += 1.0
            elif index == total // 2:
                score += 1.0

            # Small penalty for very narrative / structural chunks dominated by
            # section-like language; we still allow them but make them less likely
            # to be selected when more concept-dense chunks exist.
            structural_tokens = [
                "introduction",
                "background",
                "related work",
                "literature review",
                "organization of this paper",
                "in this section",
                "in the next section",
                "the rest of this paper",
                "section",
                "chapter",
            ]
            for st in structural_tokens:
                if st in t:
                    score -= 0.8

            return score

        def build_micro_summary(chunks: List[str]) -> str:
            """Construct a short, dense summary paragraph from top chunks."""
            if not chunks:
                return ""

            total = len(chunks)
            scored = [
                (score_chunk(text or "", idx, total), idx, text or "")
                for idx, text in enumerate(chunks)
            ]

            # Sort by score descending, then by original order
            scored.sort(key=lambda x: (-x[0], x[1]))

            selected_texts: List[str] = []
            for score, _, text in scored[:3]:  # take up to top 3 chunks
                if score <= 0.0:
                    continue
                cleaned = " ".join(text.split())  # normalize whitespace
                if cleaned:
                    selected_texts.append(cleaned[:400])

            if not selected_texts:
                # Fallback: use first, middle, last if scoring found nothing
                indices = sorted(set([0, total // 2, max(total - 1, 0)]))
                for idx in indices:
                    if 0 <= idx < total:
                        cleaned = " ".join((chunks[idx] or "").split())
                        if cleaned:
                            selected_texts.append(cleaned[:400])

            if not selected_texts:
                return ""

            # Join into a single paragraph and trim to a reasonable length.
            # This is intentionally concept-dense: we keep the most informative
            # parts about phenomena, variables, tasks, datasets, equipment,
            # methodology, and conclusions rather than narrative structure.
            paragraph = " ".join(selected_texts)
            return paragraph[:1200]

        for paper in papers:
            results = self.collection.get(where={"paper_id": paper["paper_id"]})
            docs = results.get("documents") or []

            micro_summary = build_micro_summary(docs)
            if not micro_summary:
                # Fallback: just use the canonical title as summary
                micro_summary = paper.get("paper_title", paper["paper_filename"])

            summaries.append(
                {
                    "paper_id": paper["paper_id"],
                    "paper_filename": paper["paper_filename"],
                    "paper_title": paper.get("paper_title", paper["paper_filename"]),
                    "summary": micro_summary,
                }
            )

        return summaries


    def count_chunks(self) -> int:
        """Get total number of chunks in the database."""
        return self.collection.count()

    def reset(self):
        """Clear all data from the collection."""
        with self._lock:
            self.client.delete_collection(name="paper_chunks")
            self.collection = self.client.get_or_create_collection(
                name="paper_chunks",
                metadata={"description": "Chunks from journal articles"},
            )
