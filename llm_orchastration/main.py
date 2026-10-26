from orchastration.orchastrator import Orchastrator

import traceback

def main():
    try:
        orchestrator = Orchastrator()
    except Exception as exc:
        print("Failed to initialize the LLM orchestrator.")
        print(f"Error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    print("LLM Orchestrator")
    print("Type 'exit' to quit.")

    try:
        while True:
            try:
                user_query = input("\nYou: ")
            except EOFError:
                print("\nGoodbye!")
                break

            if user_query.strip().lower() == "exit":
                print("Goodbye!")
                break

            response = orchestrator.run(user_query)

            print("\nAssistant:")
            print(response)

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()