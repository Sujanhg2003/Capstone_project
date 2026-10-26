import json
from datetime import datetime , timezone
from pathlib import Path
from uuid import uuid4

LOG_DIRECTORY = Path("./logs")
LOG_FILE = LOG_DIRECTORY / "audit.jsonl"

class AuditLogger:

    def __init__(self):
        LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    def log_request(
        self,
        question: str,
        answer: str,
        sources: list[dict],
        latency: dict,
        retrieval_info: dict,
        model: str,
    ) -> str:

        request_id = str(uuid4())

        record = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),

            "question": question,

            "model": model,

            "retrieval": {
                "status": retrieval_info.get("retrieval_status"),
                "top_k": retrieval_info.get("top_k"),
                "distance_threshold": retrieval_info.get(
                    "distance_threshold"
                ),
                "chunks_returned": retrieval_info.get(
                    "chunks_returned"
                ),
            },

            "sources": sources,

            "answer": answer,

            "latency_ms": {
                key: value
                for key, value in latency.items()
                if isinstance(value, (int, float))
            },
        }

        with LOG_FILE.open(
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

        return request_id