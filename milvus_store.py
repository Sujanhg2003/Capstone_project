import requests
from pymilvus import MilvusClient , DataType

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text:latest"

MILVUS_DB = "milvus_demo.db"
COLLECTION_NAME = "documents"

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

client = MilvusClient(
    uri=MILVUS_DB
)

print("Connected to Milvus")


documents = [
    "Machine learning is a branch of artificial intelligence that learns patterns from data.",

    "Deep learning uses neural networks with multiple layers to learn complex patterns.",

    "Retrieval Augmented Generation combines information retrieval with large language models.",

    "Vector databases store embeddings and perform similarity search over high-dimensional vectors."
]


first_embedding = get_embedding(
    documents[0]
)

embedding_dimension = len(first_embedding)
print("Embedding dimension:", embedding_dimension)



if client.has_collection(COLLECTION_NAME):

    client.drop_collection(COLLECTION_NAME)


schema = client.create_schema(
    auto_id=False,
    enable_dynamic_field=False
)


schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True
)

schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=5000
)

schema.add_field(
    field_name="embedding",
    datatype=DataType.FLOAT_VECTOR,
    dim=embedding_dimension
)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)

print("Collection created")


data = []

for i, document in enumerate(documents):

    embedding = get_embedding(document)

    data.append({
        "id": i,
        "text": document,
        "embedding": embedding
    })

    print(
        f"Generated embedding for document {i}"
    )

result = client.insert(
    collection_name=COLLECTION_NAME,
    data=data
)

print("\nInserted documents:")
print(result)

client.flush(
    collection_name=COLLECTION_NAME
)

print("\nMilvus database ready!")

print(
    "Entities:",
    client.get_collection_stats(
        collection_name=COLLECTION_NAME
    )
)