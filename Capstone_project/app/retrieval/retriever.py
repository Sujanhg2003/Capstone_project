from app.vectorstore.embedder import OllamaEmbedder
from app.vectorstore.chroma_store import ChromaStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import CohereReranker

from app.config import (
    SEMANTIC_TOP_K,
    KEYWORD_TOP_K,
    RERANK_TOP_K,
    RRF_K
)

class HybridRetriever:

    def __init__(
        self,
        semantic_top_k: int = SEMANTIC_TOP_K,
        keyword_top_k: int = KEYWORD_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        rrf_k: int = RRF_K
    ):
        self.semantic_top_k = semantic_top_k
        self.keyword_top_k = keyword_top_k
        self.rerank_top_k = rerank_top_k
        self.rrf_k = rrf_k

        self.embedder = OllamaEmbedder()

        self.vector_store = ChromaStore()

        # Load corpus from ChromaDB
        chunks = self.vector_store.get_all_chunks()

        self.bm25 = BM25Retriever(
            chunks
        )

        self.reranker = CohereReranker()

    def semantic_search(
        self,
        query: str
    ) -> list[dict]:

        query_embedding = (
            self.embedder.embed_text(query)
        )

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.semantic_top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        results_list = []

        for i in range(len(documents)):

            results_list.append({

                "chunk_id": ids[i],

                "text": documents[i],

                "metadata": metadatas[i],

                "vector_distance": float(
                    distances[i]
                )
            })

        return results_list

    def hybrid_search(
        self,
        query: str
    ) -> list[dict]:

        semantic_results = (
            self.semantic_search(query)
        )

        keyword_results = (
            self.bm25.search(
                query,
                self.keyword_top_k
            )
        )

        fused = {}

        for rank, result in enumerate(
            semantic_results,
            start=1
        ):

            chunk_id = result["chunk_id"]

            fused.setdefault(
                chunk_id,
                {
                    "document": result,
                    "rrf_score": 0.0
                }
            )

            fused[chunk_id][
                "rrf_score"
            ] += 1 / (
                self.rrf_k + rank
            )

        for rank, result in enumerate(
            keyword_results,
            start=1
        ):

            chunk_id = result["chunk_id"]

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "document": result,
                    "rrf_score": 0.0
                }

            fused[chunk_id][
                "rrf_score"
            ] += 1 / (
                self.rrf_k + rank
            )

        ranked = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        candidates = []

        for item in ranked[:10]:

            document = item[
                "document"
            ].copy()

            document[
                "rrf_score"
            ] = item["rrf_score"]

            candidates.append(
                document
            )

        return candidates

    def retrieve(
        self,
        query: str
    ) -> list[dict]:

        candidates = self.hybrid_search(
            query
        )

        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=self.rerank_top_k,
            score_threshold= 0.3
        )

        return reranked

    