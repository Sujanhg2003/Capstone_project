import requests
import chromadb


url = "http://localhost:11434/api/embed"

CHROMA_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name="my_documents"
)


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

res = {}

for i , document in enumerate(documents):
    payload = {
        "model":"nomic-embed-text",
        "input":document
    }

    response = requests.post(url,json=payload)

    res[f"doc_{i}"] = response.json()["embeddings"]




print(res.keys())
print(res['doc_0'])
print("len:",len(res['doc_0'][0]))

# payload = {
#     "model": "nomic-embed-text",
#     "input": "RAG combines retrieval with generation."
# }

# response = requests.post(url, json=payload)

# print(response.json())