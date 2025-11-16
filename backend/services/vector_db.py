"""Vector database service for semantic search using ChromaDB."""

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
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number or 0,
            }
            for chunk in chunks
        ]

        # Add to collection (ChromaDB handles embedding automatically)
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
            List of dictionaries with paper_id and paper_filename
        """
        # Get all metadata
        results = self.collection.get()
        
        # Extract unique papers
        papers = {}
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                paper_id = metadata.get("paper_id")
                paper_filename = metadata.get("paper_filename")
                if paper_id and paper_filename:
                    papers[paper_id] = paper_filename
        
        return [{"paper_id": pid, "paper_filename": fname} for pid, fname in papers.items()]

    def get_paper_summaries(self) -> List[dict]:
        """
        Get lightweight summaries of all papers (first chunk only - usually contains title/abstract).

        Returns:
            List of dictionaries with paper_id, filename, and summary content
        """
        papers = self.get_all_papers()
        summaries = []
        
        for paper in papers:
            # Get only the first chunk (chunk_index=0) which typically has title/abstract
            # ChromaDB requires $and operator for multiple conditions
            results = self.collection.get(
                where={
                    "$and": [
                        {"paper_id": paper["paper_id"]},
                        {"chunk_index": 0}
                    ]
                },
                limit=1
            )
            
            if results["ids"] and results["documents"]:
                summaries.append({
                    "paper_id": paper["paper_id"],
                    "paper_filename": paper["paper_filename"],
                    "summary": results["documents"][0][:500]  # First 500 chars
                })
            else:
                # Fallback: get any chunk from this paper
                results = self.collection.get(
                    where={"paper_id": paper["paper_id"]},
                    limit=1
                )
                if results["ids"] and results["documents"]:
                    summaries.append({
                        "paper_id": paper["paper_id"],
                        "paper_filename": paper["paper_filename"],
                        "summary": results["documents"][0][:500]
                    })
                else:
                    # Last fallback: just the filename
                    summaries.append({
                        "paper_id": paper["paper_id"],
                        "paper_filename": paper["paper_filename"],
                        "summary": paper["paper_filename"]
                    })
        
        return summaries


    def count_chunks(self) -> int:
        """Get total number of chunks in the database."""
        return self.collection.count()

    def reset(self):
        """Clear all data from the collection."""
        self.client.delete_collection(name="paper_chunks")
        self.collection = self.client.get_or_create_collection(
            name="paper_chunks",
            metadata={"description": "Chunks from journal articles"},
        )
