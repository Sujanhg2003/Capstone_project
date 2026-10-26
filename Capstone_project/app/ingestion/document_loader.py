from pathlib import Path
from pypdf import PdfReader

def load_pdf(file_path: str) ->dict:
    """
    Load a single PDF and extract its text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected PDF file, got: {path.suffix}"
        )


    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        pages.append({
            "page_number": page_number,
            "text": text
        })

    full_text = "\n\n".join(
        page["text"]
        for page in pages
    )

    return {
        "file_name": path.name,
        "file_path": str(path),
        "total_pages": len(reader.pages),
        "text": full_text,
        "pages": pages
    }

def load_all_pdfs(directory: str) -> list[dict]:
    """
    Load all PDF files from a directory.
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    pdf_files = sorted(
        directory_path.glob("*.pdf")
    )

    documents = []

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        document = load_pdf(str(pdf_file))

        documents.append(document)

    return documents