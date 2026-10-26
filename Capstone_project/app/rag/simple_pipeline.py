import time

from app.retrieval.simple_retriever import SimpleRetriever
from app.ollama_client import OllamaLLM
from app.rag.prompt import (
    SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    build_standalone_question_prompt,
    build_user_prompt
)

from app.logging.audit_logger import AuditLogger
from app.config import LLM_MODEL

class SimpleRAGPipeline:

    def __init__(self):

        self.retriever = SimpleRetriever(
            top_k=3
        )
        self.llm = OllamaLLM()
        self.audit_logger  =AuditLogger()

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> dict:

        total_start = time.perf_counter()
        history = history or []
        question = question.strip()

        retrieval_question = question
        query_rewrite_ms = 0.0

        if history:
            rewrite_prompt = build_standalone_question_prompt(
                question=question,
                history=history,
            )

            rewrite_start = time.perf_counter()

            rewritten_question = self.llm.generate(
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                user_prompt=rewrite_prompt,
            )

            query_rewrite_ms = (
                time.perf_counter() - rewrite_start
            ) * 1000

            if (
                isinstance(rewritten_question, str)
                and rewritten_question.strip()
            ):
                retrieval_question = rewritten_question.strip()


        results, retrieval_info = self.retriever.retrieve(
            retrieval_question
        )

        if not results:

            answer = (
                "I don't have enough relevant information in "
                "the provided company documents to answer "
                "this question."
            )

            total_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            latency = {
                key: value
                for key, value in retrieval_info.items()
                if key.endswith("_ms")
            }

            latency.update({
                "query_rewrite_ms": query_rewrite_ms,
                "prompt_ms": 0.0,
                "llm_ms": 0.0,
                "total_ms": total_ms,
            })

            request_id = self.audit_logger.log_request(
                question=question,
                answer=answer,
                sources=[],
                latency=latency,
                retrieval_info={
                    **retrieval_info,
                    "retrieval_question": retrieval_question,
                },
                model=LLM_MODEL,
            )

            return {
                "request_id": request_id,
                "question": question,
                "retrieval_question": retrieval_question,
                "answer": answer,
                "sources": [],
                "results": [],
                "latency": latency,
            }



        prompt_start = time.perf_counter()

        user_prompt = build_user_prompt(
            question=question,
            results=results,
            history=history,
            retrieval_question=retrieval_question,
        )

        prompt_ms = (
            time.perf_counter() - prompt_start
        ) * 1000


        llm_start = time.perf_counter()

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        llm_ms = (
            time.perf_counter() - llm_start
        ) * 1000


        sources = []
        seen_sources = set()

        for result in results:

            metadata = result["metadata"]
            source = metadata["source"]
            page = metadata["page"]

            source_key = (
                source,
                page
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            sources.append({
                "document": source,
                "page": page
            })


        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        latency = {
            "embedding_ms": retrieval_info["embedding_ms"],
            "chroma_ms": retrieval_info["chroma_ms"],
            "retrieval_total_ms": (
                retrieval_info["retrieval_total_ms"]
            ),
            "query_rewrite_ms": query_rewrite_ms,
            "prompt_ms": prompt_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
        }

        request_id = self.audit_logger.log_request(
            question=question,
            answer=answer,
            sources=sources,
            latency=latency,
            retrieval_info={
                **retrieval_info,
                "retrieval_question": retrieval_question,
            },
            model=LLM_MODEL,
        )

        return {
            "request_id": request_id,
            "question": question,
            "retrieval_question": retrieval_question,
            "answer": answer,
            "sources": sources,
            "results": results,
            "latency": latency,
        }