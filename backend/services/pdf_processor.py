"""PDF processing service for extracting text and metadata from papers."""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import google.generativeai as genai

from backend.core.config import settings
from backend.models.schemas import PaperMetadata, PaperChunk


class PDFProcessor:
    """Service for processing PDF files."""

    def __init__(self):
        """Initialize PDF processor."""
        # Configure the LLM lazily; this is cheap and reused across calls.
        if settings.google_api_key:
            genai.configure(api_key=settings.google_api_key)
            model_name = getattr(settings.agent, "orchestrator_model", None) or settings.agent.model
            
            # Configure safety settings to be permissive for document analysis
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
            
            self._title_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 1024,
                },
                safety_settings=safety_settings,
            )
        else:
            self._title_model = None

    def _infer_title_with_ai(self, text: str) -> str:
        """Use an LLM to infer the paper title from early text.

        Returns an empty string if inference is not possible or disabled.
        """
        if not self._title_model:
            return ""

        full_text = (text or "").strip()
        if not full_text:
            return ""

        # Get config values
        min_len = getattr(settings.chunking, "min_title_snippet_chars", 2000)
        max_len = getattr(settings.chunking, "title_snippet_chars", 10000)
        
        # Define attempts: first try small snippet, then max snippet if needed
        attempts = []
        
        # If text is short, just one attempt with full text
        if len(full_text) <= min_len:
            attempts.append(full_text)
        else:
            # First attempt: min_len
            attempts.append(full_text[:min_len])
            # Second attempt: max_len (if we have more text than min_len)
            if len(full_text) > min_len:
                attempts.append(full_text[:max_len])

        for i, snippet in enumerate(attempts):
            # If this is a retry (i > 0), we might want to log it or just proceed
            
            prompt = (
                "Extract the MAIN TITLE of the research paper from the text below.\n"
                "WARNING: The text extraction might be out of order. The title might appear AFTER the 'Introduction' or 'Keywords'.\n"
                "It might even appear at the very END of the text block.\n"
                "Look for a standalone phrase that sounds like a research topic, often near 'Conference' or author names.\n"
                "Ignore 'Procedia', 'Elsevier', 'ScienceDirect', dates, and volume numbers.\n"
                "Return ONLY the title text.\n\n"
                f"Text:\n{snippet}"
            )

            try:
                response = self._title_model.generate_content(prompt)
                try:
                    raw = (response.text or "").strip()
                except Exception:
                    # Fallback if .text accessor fails (e.g. blocked response)
                    candidates = getattr(response, "candidates", []) or []
                    parts = []
                    for cand in candidates:
                        content = getattr(cand, "content", None)
                        if content and getattr(content, "parts", None):
                            for part in content.parts:
                                if getattr(part, "text", None):
                                    parts.append(part.text)
                    raw = " ".join(parts).strip()
            except Exception:
                continue

            # Basic sanity checks: non-trivial, contains letters, reasonable length
            if not raw or len(raw) < 5 or len(raw) > 400:
                continue
            if not any(c.isalpha() for c in raw):
                continue
                
            # Filter out common false positives (headers/journal names inferred as titles)
            lower_raw = raw.lower()
            if lower_raw.startswith("vol ") or lower_raw.startswith("vol.") or "issn" in lower_raw:
                continue
                
            # If we got a valid-looking title, return it immediately
            return " ".join(raw.split())

        # If all attempts failed
        return ""

    def build_canonical_title(self, filename: str, inferred_title: str) -> str:
        """
        Determine the canonical title.
        
        Per user requirement:
        - If inferred_title is available, use it.
        - Else, use filename.
        """
        if inferred_title and inferred_title.strip():
            return inferred_title.strip()
        return filename

    def extract_metadata(self, pdf_path: Path, original_filename: str | None = None) -> PaperMetadata:
        """Extract metadata from a PDF file."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Generate unique ID based on file content
        paper_id = self._generate_paper_id(pdf_path)

        # Get file stats
        stat = pdf_path.stat()
        file_size = stat.st_size
        created_at = datetime.fromtimestamp(stat.st_ctime)

        # Open PDF to get page count and text snippet for title inference
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)

            # Start with any embedded document title
            doc_title = (doc.metadata or {}).get("title") or ""

            # Fallback snippet: first page text
            first_page_text = ""
            try:
                first_page_text = (doc[0].get_text() or "").strip()
            except Exception:
                first_page_text = ""

        # Decide on logical filename for display: prefer the original upload name
        # if provided, otherwise fall back to the on-disk name.
        logical_filename = original_filename or pdf_path.name

        # Use AI plus PDF metadata to infer a human-readable title
        inferred_title = doc_title.strip()
        
        # Ignore generic or useless metadata titles so we fall back to AI
        # Use exact matches or startswith to avoid false positives (e.g. "presentation" in "representation")
        bad_titles_exact = {
            "untitled", "presentation", "document", "microsoft word", 
            "unknown", "context unknown", "pdf", "converted"
        }
        
        title_lower = inferred_title.lower().strip()
        
        # Check for exact matches or specific prefixes
        is_bad = (
            title_lower in bad_titles_exact or
            title_lower.startswith("microsoft word -") or
            title_lower.startswith("vol ") or  # "Vol 12"
            title_lower.startswith("vol.") or
            "cet vol" in title_lower  # Specific case mentioned by user
        )
        
        if is_bad:
            inferred_title = ""
            
        # Also ignore titles that look like filenames or are too short/numeric
        if len(inferred_title) < 5 or inferred_title.lower().endswith(".pdf"):
            inferred_title = ""

        if not inferred_title:
            inferred_title = self._infer_title_with_ai(first_page_text)

        # Build canonical title using heuristic to avoid redundant duplication
        canonical_title = self.build_canonical_title(logical_filename, inferred_title)

        return PaperMetadata(
            id=paper_id,
            filename=logical_filename,
            canonical_title=canonical_title,
            inferred_title=inferred_title,
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
        self,
        pdf_path: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        original_filename: str | None = None,
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
        # Extract metadata (using the original upload filename for display)
        metadata = self.extract_metadata(pdf_path, original_filename=original_filename)

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

