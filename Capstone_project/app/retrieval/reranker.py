import cohere

from app.config import (
    COHERE_API_KEY , COHERE_RERANK_MODEL
    )

class CohereReranker:

    def __init__(self):
        if not COHERE_API_KEY:
            raise ValueError(
                "COHERE_API_KEY is not configured."
            )

        self.client = cohere.ClientV2(
            api_key=COHERE_API_KEY
        )

        self.model = COHERE_RERANK_MODEL

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
        score_threshold: float = 0.30
    ) -> list[dict]:

        if not documents:
            return []

        texts = [
            document["text"]
            for document in documents
        ]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=min(
                top_k,
                len(texts)
            )
        )

        results = []

        for result in response.results:

            score = float(
                result.relevance_score
            )

            if score < score_threshold:
                continue


            original_document = documents[
                result.index
            ].copy()

            original_document[
                "rerank_score"
            ] = float(
                result.relevance_score
            )

            results.append(
                original_document
            )

        return results

