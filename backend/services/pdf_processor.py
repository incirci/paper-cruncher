"""PDF processing service for extracting text and metadata from papers."""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from backend.models.schemas import PaperMetadata, PaperChunk


class PDFProcessor:
    """Service for processing PDF files."""

    def __init__(self):
        """Initialize PDF processor."""
        pass

    def extract_metadata(self, pdf_path: Path) -> PaperMetadata:
        """Extract metadata from a PDF file."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Generate unique ID based on file content
        paper_id = self._generate_paper_id(pdf_path)

        # Get file stats
        stat = pdf_path.stat()
        file_size = stat.st_size
        created_at = datetime.fromtimestamp(stat.st_ctime)

        # Open PDF to get page count
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)

        return PaperMetadata(
            id=paper_id,
            filename=pdf_path.name,
            filepath=str(pdf_path),
            page_count=page_count,
            file_size=file_size,
            created_at=created_at,
            indexed_at=datetime.now(),
        )

    def extract_text(self, pdf_path: Path) -> str:
        """Extract all text from a PDF file."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text_parts = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())

        return "\n\n".join(text_parts)

    def extract_text_with_pages(self, pdf_path: Path) -> List[tuple[int, str]]:
        """Extract text from PDF with page numbers."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pages_text = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():  # Only include pages with text
                    pages_text.append((page_num, text))

        return pages_text

    def process_paper(
        self, pdf_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> tuple[PaperMetadata, List[PaperChunk]]:
        """
        Process a PDF paper: extract metadata and create chunks.

        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters

        Returns:
            Tuple of (metadata, chunks)
        """
        # Extract metadata
        metadata = self.extract_metadata(pdf_path)

        # Extract text with page numbers
        pages_text = self.extract_text_with_pages(pdf_path)

        # Create chunks
        chunks = []
        chunk_index = 0

        for page_num, page_text in pages_text:
            # Split page text into chunks
            page_chunks = self._create_chunks(
                page_text, chunk_size, chunk_overlap, metadata.id, page_num, chunk_index
            )
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        return metadata, chunks

    def _create_chunks(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        paper_id: str,
        page_num: int,
        start_index: int,
    ) -> List[PaperChunk]:
        """Create overlapping chunks from text."""
        chunks = []
        start = 0
        chunk_index = start_index

        while start < len(text):
            # Get chunk text
            end = start + chunk_size
            chunk_text = text[start:end]

            # Skip empty chunks
            if chunk_text.strip():
                chunk_id = f"{paper_id}_chunk_{chunk_index}"
                chunks.append(
                    PaperChunk(
                        id=chunk_id,
                        paper_id=paper_id,
                        content=chunk_text,
                        chunk_index=chunk_index,
                        page_number=page_num,
                    )
                )
                chunk_index += 1

            # Move to next chunk with overlap
            start = end - chunk_overlap if end < len(text) else end

        return chunks

    def _generate_paper_id(self, pdf_path: Path) -> str:
        """Generate a unique ID for a paper based on its content hash."""
        hasher = hashlib.sha256()

        # Hash file content
        with open(pdf_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()[:16]

    def scan_papers_folder(self, folder_path: Path) -> List[Path]:
        """Scan folder for PDF files."""
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            return []

        return list(folder_path.glob("*.pdf"))
