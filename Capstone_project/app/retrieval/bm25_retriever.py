from rank_bm25 import BM25Okapi

class BM25Retriever:

    def __init__(self, chunks: list[dict]):

        self.chunks = chunks

        corpus = [
            chunk["text"]
            for chunk in chunks
        ]

        tokenized_corpus = [
            self.tokenize(text)
            for text in corpus
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:

        return text.lower().split()

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> list[dict]:

        tokenized_query = self.tokenize(
            query
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            result = self.chunks[index].copy()

            result["bm25_score"] = float(
                scores[index]
            )

            results.append(result)

        return results