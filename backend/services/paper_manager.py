"""Paper management service for indexing and managing papers."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from backend.core.config import settings
from backend.models.schemas import PaperMetadata
from backend.services.pdf_processor import PDFProcessor
from backend.services.vector_db import VectorDBService


class PaperManager:
    """Service for managing papers and their indexing."""

    def __init__(self, pdf_processor: PDFProcessor, vector_db: VectorDBService):
        """
        Initialize paper manager.

        Args:
            pdf_processor: PDF processing service
            vector_db: Vector database service
        """
        self.pdf_processor = pdf_processor
        self.vector_db = vector_db
        self.papers: Dict[str, PaperMetadata] = {}
        self.uploads_dir = settings.get_vector_db_path().parent / "uploads"
        self.metadata_file = settings.get_vector_db_path() / "papers_metadata.json"

        # Load existing paper metadata from disk
        self._load_metadata()

    def _load_metadata(self):
        """Load paper metadata from disk, limited to uploads.

        Any metadata entries whose underlying file path is not inside the
        uploads directory are skipped. This effectively drops legacy
        folder-based papers while keeping only uploaded PDFs.
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for paper_data in data:
                        metadata = PaperMetadata(**paper_data)

                        # Check if file exists in local uploads dir (handles Docker vs Local path mismatch)
                        if metadata.filename:
                            local_path = self.uploads_dir / metadata.filename
                            if local_path.exists():
                                metadata.filepath = str(local_path)
                                self.papers[metadata.id] = metadata
                                continue

                        file_path = getattr(metadata, "filepath", None)
                        if not file_path:
                            continue

                        path_obj = Path(file_path)
                        try:
                            # Only keep papers under uploads_dir
                            if not path_obj.is_absolute():
                                path_obj = (self.uploads_dir / path_obj).resolve()
                            if self.uploads_dir in path_obj.parents:
                                self.papers[metadata.id] = metadata
                        except Exception:
                            # If resolution fails, skip this entry
                            continue
            except Exception as e:
                print(f"Error loading paper metadata: {e}")

    def _save_metadata(self):
        """Save paper metadata to disk."""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            papers_list = [paper.model_dump() for paper in self.papers.values()]
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(papers_list, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving paper metadata: {e}")

    def _index_paper(self, pdf_path: Path, original_filename: str | None = None):
        """Index a single paper."""
        # Process paper
        metadata, chunks = self.pdf_processor.process_paper(
            pdf_path,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            original_filename=original_filename,
        )

        # Add to vector database
        self.vector_db.add_paper_chunks(chunks, metadata)

        # Store metadata
        self.papers[metadata.id] = metadata
        
        # Save metadata to disk
        self._save_metadata()

    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get paper metadata by ID."""
        return self.papers.get(paper_id)

    def list_papers(self) -> List[PaperMetadata]:
        """Get list of all papers (uploads only)."""
        return list(self.papers.values())

    def add_paper(self, pdf_path: Path, original_filename: str | None = None) -> PaperMetadata:
        """
        Add a new paper to the collection.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Paper metadata
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Fast path: if this exact filepath is already known, reuse existing metadata
        for meta in self.papers.values():
            try:
                if Path(meta.filepath) == pdf_path:
                    return meta
            except Exception:
                continue

        # Otherwise index the paper (this will populate self.papers and persist metadata)
        self._index_paper(pdf_path, original_filename=original_filename)

        # Return the metadata we just stored
        # We rely on PDFProcessor to assign a stable id for this path,
        # so the last entry with matching filename is a good approximation.
        filename = pdf_path.name
        for meta in reversed(list(self.papers.values())):
            if meta.filename == filename:
                return meta

        # Fallback: extract fresh metadata (does not re-index)
        return self.pdf_processor.extract_metadata(pdf_path)

    def remove_paper(self, paper_id: str):
        """
        Remove a paper from the collection.

        Args:
            paper_id: Paper ID to remove
        """
        # Remove from vector database
        self.vector_db.delete_paper(paper_id)

        # Remove from papers dict
        if paper_id in self.papers:
            del self.papers[paper_id]
            
        # Save updated metadata
        self._save_metadata()

    def reindex_all(self):
        """Reindex all known papers from metadata.

        This rebuilds the vector database using the persisted
        `papers_metadata.json` entries (including uploaded papers),
        instead of scanning a filesystem folder.
        """
        # Reload existing metadata from disk as source of truth
        self.papers.clear()
        self._load_metadata()

        # Clear vector DB and re-add all papers using stored paths
        self.vector_db.reset()
        for metadata in list(self.papers.values()):
            try:
                # Use correct attribute 'filepath'
                pdf_path = Path(metadata.filepath) if getattr(metadata, "filepath", None) else None
                
                # Fallback: try to construct path from uploads dir if absolute path is missing/wrong
                if not pdf_path or not pdf_path.exists():
                    if metadata.filename:
                        pdf_path = self.uploads_dir / metadata.filename

                if pdf_path and pdf_path.exists():
                    # Re-process and re-add chunks
                    md, chunks = self.pdf_processor.process_paper(
                        pdf_path,
                        chunk_size=settings.chunking.chunk_size,
                        chunk_overlap=settings.chunking.chunk_overlap,
                    )
                    self.vector_db.add_paper_chunks(chunks, md)
                    
                    # CRITICAL: Update the in-memory metadata with the newly inferred title/info
                    self.papers[md.id] = md
            except Exception as e:
                print(f"Error reindexing paper {metadata.id}: {e}")

        # Save metadata back
        self._save_metadata()

    def get_papers_count(self) -> int:
        """Get total number of papers."""
        return len(self.papers)

    def reset_all(self) -> None:
        """Clear all paper metadata and on-disk metadata file.

        Used by the admin reset endpoint so that `/api/papers` returns
        an empty list immediately after a reset, without relying on
        stale in-memory state.
        """
        self.papers.clear()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
