from typing import List

from app.config import CHUNK_SIZE  ,CHUNK_OVERLAP

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:

    if not text or not text.strip():
        return []

    text = text.strip()

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks

def create_chunks(documents: list[dict]) -> list[dict]:

    all_chunks = []

    for document in documents:

        file_name = document["file_name"]

        for page in document["pages"]:

            page_number = page["page_number"]
            page_text = page["text"]

            chunks = chunk_text(
                text=page_text,
                chunk_size= CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )

            for index, chunk in enumerate(chunks):

                chunk_id = (
                    f"{file_name}"
                    f"_p{page_number}"
                    f"_c{index + 1}"
                )

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "source": file_name,
                        "page": page_number,
                        "chunk_index": index + 1
                    }
                })

    return all_chunks

