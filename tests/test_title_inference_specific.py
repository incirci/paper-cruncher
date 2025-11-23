
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from backend.services.pdf_processor import PDFProcessor
from backend.core.config import settings

def test_inference():
    print(f"API Key present: {bool(settings.google_api_key)}")
    
    processor = PDFProcessor()
    pdf_path = Path("data/uploads/SISTA-09-134.pdf")
    
    if not pdf_path.exists():
        print("File not found")
        return

    print(f"Processing {pdf_path}")
    
    # Manually call the internal methods to debug
    with open(pdf_path, "rb") as f:
        import fitz
        with fitz.open(pdf_path) as doc:
            print(f"PDF Metadata: {doc.metadata}")
            text = doc[0].get_text()
            print(f"Text length: {len(text)}")
            print("-" * 40)
            print(f"Full text passed to AI:\n{text}")
            print("-" * 40)
            
            # Test AI inference directly
            print("Calling AI inference...")
            inferred = processor._infer_title_with_ai(text)
            print(f"AI returned: '{inferred}'")
            
            # Test bad title logic (reproducing logic from pdf_processor.py)
            bad_titles_exact = {
                "untitled", "presentation", "document", "microsoft word", 
                "unknown", "context unknown", "pdf", "converted"
            }
            title_lower = inferred.lower().strip()
            is_bad = (
                title_lower in bad_titles_exact or
                title_lower.startswith("microsoft word -") or
                title_lower.startswith("vol ") or
                title_lower.startswith("vol.") or
                "cet vol" in title_lower
            )
            print(f"Is bad title (pre-check): {is_bad}")

            # Test post-inference filters (reproducing logic from _infer_title_with_ai)
            raw = inferred
            lower_raw = raw.lower()
            is_filtered = False
            if not raw or len(raw) < 5 or len(raw) > 400:
                is_filtered = True
                print("Filtered by length")
            elif not any(c.isalpha() for c in raw):
                is_filtered = True
                print("Filtered by alpha check")
            elif lower_raw.startswith("vol ") or lower_raw.startswith("vol.") or "issn" in lower_raw or "doi:" in lower_raw:
                is_filtered = True
                print("Filtered by content (vol/issn/doi)")
            
            print(f"Is filtered post-inference: {is_filtered}")
            
            # Test metadata extraction
            meta = processor.extract_metadata(pdf_path)
            print(f"Final Canonical Title: {meta.canonical_title}")

if __name__ == "__main__":
    test_inference()
