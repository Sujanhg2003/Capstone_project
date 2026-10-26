import time

from app.vectorstore.embedder import OllamaEmbedder
from app.vectorstore.chroma_store import ChromaStore

from app.config import TOP_K , RETRIEVAL_DISTANCE_THRESHOLD

class SimpleRetriever:
    def __init__(self, top_k: int= TOP_K,
                 distance_threshold:float = RETRIEVAL_DISTANCE_THRESHOLD):
        self.top_k = top_k
        self.embedder = OllamaEmbedder()
        self.vector_store = ChromaStore()
        self.distance_threshold = distance_threshold


    def retrieve(
            self,
            query: str
    ) -> tuple[list[dict],dict]:
        total_start = time.perf_counter()

        # -----------------------------
        # Query embedding
        # -----------------------------

        start = time.perf_counter()

        query_embedding = (
            self.embedder.embed_text(query)
        )

        embedding_ms = (
            time.perf_counter() - start
        ) * 1000

        # -----------------------------
        # ChromaDB search
        # -----------------------------

        start = time.perf_counter()

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )

        chroma_ms = (
            time.perf_counter() - start
        ) * 1000

        # -----------------------------
        # Format results
        # -----------------------------

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        retrieved_chunks = []

        for i in range(len(documents)):

            distance = float(distances[i])

            if distance > self.distance_threshold:
                continue

            retrieved_chunks.append({

                "chunk_id": ids[i],

                "text": documents[i],

                "metadata": metadatas[i],

                "distance": distance
            })

        if retrieved_chunks:

            retrieval_status = "accepted"

        else:

            retrieval_status = "rejected"

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        latency = {
            "embedding_ms": embedding_ms,
            "chroma_ms": chroma_ms,
            "retrieval_total_ms": total_ms
        }

        retrieval_info = {
            
            "retrieval_status": retrieval_status,
            "distance_threshold": self.distance_threshold,
            "chunks_returned": len(retrieved_chunks),
            "top_k":self.top_k
        }

        return (
            retrieved_chunks,
            {
                **latency,
                **retrieval_info
            }
        )