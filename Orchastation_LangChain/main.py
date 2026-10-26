from graph.workflow import graph

while True:
    user_input = input("\nYou:")

    if user_input.lower() == "exit":
            print("Goodbye!")
            break

    result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            }
        )

    print("\nAssistant:")
    print(
            result["messages"][-1].content
        )



