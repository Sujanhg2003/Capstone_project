
from app.rag.simple_pipeline import SimpleRAGPipeline


from app.memory.conversation_memory import (
    ConversationMemory
)

memory = ConversationMemory()

def main():
    rag = SimpleRAGPipeline()
    session_id = "console-session"

    print("\nInternal Knowledge Assistant")
    print("Type 'exit' to quit.")
    print("Type '/clear' to clear conversation memory.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        if question.lower() == "/clear":
            memory.clear(session_id)
            print("\nAssistant: Conversation memory cleared.")
            continue

        # Load history BEFORE processing the current question.
        history = memory.get_history(session_id)

        result = rag.ask(
            question=question,
            history=history,
        )

        print("\nAssistant:")
        print(result["answer"])

        print("\nSources:")

        if result["sources"]:
            for source in result["sources"]:
                print(
                    f"- {source['document']} "
                    f"(Page {source['page']})"
                )
        else:
            print("- No supporting document found.")

        print(f"\nRequest ID: {result['request_id']}")

        print("\nLatency:")

        for key, value in result["latency"].items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f} ms")
            else:
                print(f"{key}: {value}")

        # Save both successful answers and refusals.
        memory.add_turn(
            session_id=session_id,
            question=question,
            answer=result["answer"],
        )


if __name__ == "__main__":
    main()