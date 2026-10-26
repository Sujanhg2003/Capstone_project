import requests
from pymilvus import MilvusClient

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text:latest"

MILVUS_DB = "milvus_demo.db"
COLLECTION_NAME = "documents"

client = MilvusClient(
    uri=MILVUS_DB
)

print("Connected to Milvus")

client.load_collection(
    collection_name=COLLECTION_NAME
)

print("Collection loaded")

def get_embedding(text: str):

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
            "Ollama returned empty embedding"
        )

    return embeddings[0]


def retrieve(query,top_k = 3):

    query_embedding = get_embedding(query)
    print(
        "Query embedding dimension:",
        len(query_embedding)
    )

    results = client.search(
        collection_name=COLLECTION_NAME,

        data=[query_embedding],

        limit=top_k,

        output_fields=[
            "id",
            "text"
        ]
    )

    retrieved_documents = []

    for result in results[0]:

        retrieved_documents.append({
            "id": result["id"],
            "text": result["entity"]["text"],
            "score": result["distance"]
        })


    return retrieved_documents



query = "What is retrieval augmented generation?"

results = retrieve(
    query,
    top_k=2
)

print("\nQuery:")
print(query)

print("\nRetrieved documents:")

for result in results:

    print("\n-----------------------------")

    print("ID:", result["id"])

    print("Score:", result["score"])

    print("Document:")
    print(result["text"])





