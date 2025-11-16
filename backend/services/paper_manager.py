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
        self.papers_folder = settings.get_papers_folder_path()
        self.metadata_file = settings.get_vector_db_path() / "papers_metadata.json"

        # Load existing paper metadata from disk
        self._load_metadata()

    def _load_metadata(self):
        """Load paper metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for paper_data in data:
                        metadata = PaperMetadata(**paper_data)
                        self.papers[metadata.id] = metadata
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

    def _scan_and_index(self):
        """Scan papers folder and index all PDFs."""
        pdf_files = self.pdf_processor.scan_papers_folder(self.papers_folder)

        for pdf_path in pdf_files:
            try:
                self._index_paper(pdf_path)
            except Exception as e:
                print(f"Error indexing {pdf_path.name}: {e}")

    def _index_paper(self, pdf_path: Path):
        """Index a single paper."""
        # Process paper
        metadata, chunks = self.pdf_processor.process_paper(
            pdf_path,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
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
        """Get list of all papers."""
        return list(self.papers.values())

    def add_paper(self, pdf_path: Path) -> PaperMetadata:
        """
        Add a new paper to the collection.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Paper metadata
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Index the paper
        self._index_paper(pdf_path)

        # Return metadata
        metadata = self.pdf_processor.extract_metadata(pdf_path)
        return metadata

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
        """Reindex all papers in the folder."""
        # Clear existing data
        self.vector_db.reset()
        self.papers.clear()

        # Scan and index
        self._scan_and_index()
        
        # Save metadata
        self._save_metadata()

    def get_papers_count(self) -> int:
        """Get total number of papers."""
        return len(self.papers)
