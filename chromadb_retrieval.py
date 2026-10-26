
import requests
import chromadb

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text:latest"
CHROMA_PATH = "./chroma_db"

# Connect to existing ChromaDB
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name="my_documents"
)


# Generate embedding for the query
def get_embedding(text):
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload
    )
    response.raise_for_status()

    result = response.json()
    return result["embeddings"][0]


# Retrieve relevant documents
def retrieve_documents(query, top_k=2):

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for doc_id, document, distance in zip(
        ids, documents, distances
    ):
        print(f"\nDocument ID: {doc_id}")
        print(f"Distance: {distance:.4f}")
        print(f"Content:\n{document}")

    return documents


# Test retrieval
if __name__ == "__main__":
    query = "What is Retrieval Augmented Generation?"
    retrieved_docs = retrieve_documents(query, top_k=2)
