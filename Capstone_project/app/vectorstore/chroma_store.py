import chromadb

from app.ingestion.document_loader import load_all_pdfs
from app.ingestion.chunker import create_chunks
from app.vectorstore.embedder import OllamaEmbedder

from app.config import (
    CHROMA_PATH ,  COLLECTION_NAME
)

class ChromaStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME
            )
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ):

        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            chunk["metadata"]
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def count(self) -> int:
        return self.collection.count()


    def get_all_chunks(self) -> list[dict]:

        results = self.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        chunks = []

        for i, chunk_id in enumerate(
            results["ids"]
        ):

            chunks.append({
                "chunk_id": chunk_id,
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })

        return chunks



DOCUMENT_DIRECTORY = "./data/documents"

def main():
    print("=" * 60)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    documents = load_all_pdfs(
        DOCUMENT_DIRECTORY
    )

    print(
        f"Documents loaded: {len(documents)}"
    )

    if not documents:
        print("No PDF documents found.")
        return

    print("\n[2/4] Creating chunks...")

    chunks = create_chunks(
        documents
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )

    if not chunks:
        print("No chunks were created.")
        return

    print("\n[3/4] Generating embeddings...")

    embedder = OllamaEmbedder()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.embed_documents(
        texts
    )

    print(
        f"Embeddings generated: {len(embeddings)}"
    )

    print(
        f"Embedding dimension: {len(embeddings[0])}"
    )

    print("\n[4/4] Storing vectors in ChromaDB...")

    vector_store = ChromaStore()

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings
    )

    print(
        f"Total vectors in ChromaDB: "
        f"{vector_store.count()}"
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()

