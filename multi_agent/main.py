from graph import graph
from langchain_core.messages import HumanMessage

def main():

    user_query = input("You: ")

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_query)
            ]
        }
    )

    print("\n========== FINAL RESPONSE ==========")

    final_message = result["messages"][-1]

    print(final_message.content)


if __name__ == "__main__":
    main()