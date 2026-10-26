from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages

from langchain_ollama import  ChatOllama
from tools.calculator import calculate
from tools.datetime_tool import get_current_datetime

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

class AgentState(TypedDict):
    messages:Annotated[
        list[AnyMessage],
        add_messages
    ]

llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.3
)


tools = [
    calculate,
    get_current_datetime
]

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
    You are a tool-using assistant.

    You have access to these tools:

    1. calculate
    2. get_current_datetime

    CALCULATOR RULES:

    For addition:
    operation = "add"

    For subtraction:
    operation = "subtract"

    For multiplication:
    operation = "multiply"

    For division:
    operation = "divide"

    IMPORTANT:

    - Use the exact numbers provided by the user.
    - Do not invent numbers.
    - Do not change numbers from the user's request.
    - For an arithmetic request, call calculate exactly once.
    - After receiving a successful calculator result, do not call calculate again.
    - Use the tool result to answer the user's original question.
    - For current date/time, use get_current_datetime.
    - For general questions, do not use tools.
    - When the information is missing, say:
    "I don't have enough information in the provided context."
    """

def call_llm(state: AgentState):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
                    }
                ] + state["messages"]

    response = llm_with_tools.invoke(
        messages
    )

    print("\n========== LLM RESPONSE ==========")
    print(response)
    print("==================================\n")

    return {
        "messages": [response]
    }

tool_node = ToolNode(tools , handle_tool_errors=True)


graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "llm",
    call_llm
)

graph_builder.add_node(
    "tools",
    tool_node
)

graph_builder.add_edge(
    START,
    "llm"
)

graph_builder.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools":"tools",
        END:END
    }
)

graph_builder.add_edge(
    "tools",
    "llm"
)



graph = graph_builder.compile()