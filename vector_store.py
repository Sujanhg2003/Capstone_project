import requests
import chromadb

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text:latest"
CHROMA_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name="my_documents"
)

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

    embeddings = result["embeddings"]

    if not embeddings:
        raise ValueError("Ollama returned no embeddings")

    embedding = embeddings[0]

    if not embedding:
        raise ValueError("Ollama returned an empty embedding vector")

    print("Embedding dimension:", len(embedding))

    return embedding


documents = [
    """
    Machine learning is a branch of artificial intelligence
    that allows computers to learn patterns from data and make
    predictions without being explicitly programmed.
    """,

    """
    Deep learning is a subset of machine learning that uses
    neural networks with multiple layers to learn complex
    patterns from large amounts of data.
    """,

    """
    Retrieval Augmented Generation, commonly called RAG,
    combines information retrieval with large language models.
    Relevant documents are retrieved from a vector database
    and provided to the language model as context.
    """,

    """
    Vector databases are specialized databases designed to
    store and search high-dimensional vector embeddings.
    They are commonly used for semantic search and RAG systems.
    """
]


for i, document in enumerate(documents):

    embedding = get_embedding(document)

    print("Type:", type(embedding))
    print("Dimension:", len(embedding))
    print("First 5:", embedding[:5])

    collection.add(
        ids=[f"doc_{i}"],
        documents=[document],
        embeddings=[embedding]
    )

    print(f"Stored document {i}")

print("\nAll documents stored successfully!")