from pathlib import Path

from backend.core.config import settings
from backend.services.pdf_processor import PDFProcessor
from backend.services.paper_manager import PaperManager
from backend.services.vector_db import VectorDBService


def main() -> None:
    papers_folder = settings.get_papers_folder_path()
    db_path = settings.get_vector_db_path()

    pdf_processor = PDFProcessor()
    vector_db = VectorDBService(db_path=db_path)
    paper_manager = PaperManager(pdf_processor=pdf_processor, vector_db=vector_db)

    # 1) PDFs on disk
    pdf_files = list(papers_folder.glob("*.pdf"))
    count_pdfs_disk = len(pdf_files)

    # 2) Papers from metadata
    count_papers_metadata = paper_manager.get_papers_count()

    # 3) Papers in Chroma
    papers_in_chroma = vector_db.get_all_papers()
    count_papers_chroma = len(papers_in_chroma)

    print("Papers folder:", papers_folder)
    print("Vector DB path:", db_path)
    print("# PDFs on disk:", count_pdfs_disk)
    print("# Papers in metadata (PaperManager):", count_papers_metadata)
    print("# Unique papers in Chroma (VectorDBService):", count_papers_chroma)

    if count_pdfs_disk != count_papers_metadata:
        print("WARNING: Disk vs metadata counts differ.")
    if count_pdfs_disk != count_papers_chroma:
        print("WARNING: Disk vs Chroma counts differ.")
    if count_papers_metadata != count_papers_chroma:
        print("WARNING: Metadata vs Chroma counts differ.")


if __name__ == "__main__":
    main()
