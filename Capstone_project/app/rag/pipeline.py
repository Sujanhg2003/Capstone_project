from app.retrieval.retriever import HybridRetriever
from app.ollama_client import OllamaLLM

from app.rag.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt
)

class RAGPipeline:

    def __init__(self):
        self.retriever = HybridRetriever()

        self.llm = OllamaLLM()

    def ask(
        self,
        question: str
    ) -> dict:

        # -----------------------------
        # 1. Retrieve relevant chunks
        # -----------------------------

        results = self.retriever.retrieve(
            question
        )

        # -----------------------------
        # 2. No context
        # -----------------------------

        

        if not results:

            return {
                "question": question,
                "answer": (
                    "I don't have enough information "
                    "in the provided company documents."
                ),
                "sources": [],
                "retrieved_chunks": []
            }

        # -----------------------------
        # 3. Build prompt
        # -----------------------------

        user_prompt = build_user_prompt(
            question=question,
            results=results
        )

        # -----------------------------
        # 4. Generate answer
        # -----------------------------

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        # -----------------------------
        # 5. Collect sources
        # -----------------------------

        sources = []

        for result in results:

            sources.append({
                "document": result[
                    "metadata"
                ]["source"],

                "page": result[
                    "metadata"
                ]["page"],

                "rerank_score": result[
                    "rerank_score"
                ]
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": results
        }