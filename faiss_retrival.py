import requests
import numpy as np
import faiss
import pickle

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text:latest"

DB_PATH = "FAISS_DB"


def get_embedding(text):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "input": text
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"][0]


index = faiss.read_index(
    f"{DB_PATH}/index.faiss"
)

print("Loaded FAISS index")
print("Vectors:", index.ntotal)

with open(
    f"{DB_PATH}/documents.pkl",
    "rb"
) as f:

    documents = pickle.load(f)


print("Loaded documents:", len(documents))


query = "What is retrieval augmented generation?"

query_embedding = get_embedding(query)

query_vector = np.array(
    [query_embedding],
    dtype="float32"
)

k = 2

distances, indices = index.search(
    query_vector,
    k
)

print("\nQuery:")
print(query)

print("\nRetrieved documents:")

for distance, index_id in zip(
    distances[0],
    indices[0]
):

    print("\n-------------------------")

    print("Index:", index_id)

    print("Distance:", distance)

    print("Document:")

    print(documents[index_id])