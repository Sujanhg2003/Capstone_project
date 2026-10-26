import requests
import numpy as np
import faiss
import pickle

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text:latest"


def get_embedding(text):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    embeddings = data["embeddings"]

    if not embeddings:
        raise ValueError(
            "Ollama returned empty embeddings"
        )

    return embeddings[0]

documents = [
    """
    Machine learning is a branch of artificial intelligence
    that allows computers to learn patterns from data.
    """,

    """
    Deep learning is a subset of machine learning that uses
    neural networks with multiple layers to learn complex patterns.
    """,

    """
    Retrieval Augmented Generation combines information
    retrieval with large language models. Relevant documents
    are retrieved and provided to the LLM as context.
    """,

    """
    Vector databases store high-dimensional vector embeddings
    and allow efficient similarity search.
    """
]


embeddings = []

for document in documents:
    embedding = get_embedding(document)
    print("Embedding dimension:" , len(embedding))

    embeddings.append(embedding)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS index created")
print("Number of vectors:", index.ntotal)


print(embeddings)

print("="*60)
print(index)


faiss.write_index(
    index,
    "FAISS_DB/index.faiss"
)

with open("FAISS_DB/documents.pkl","wb") as f:
    pickle.dump(documents,f)


print("Vectors and Dcos are saved Successfully......")