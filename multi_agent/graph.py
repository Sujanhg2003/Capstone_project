from typing import TypedDict  ,Annotated

from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode

from agents.tutor_agent import TutorAgent
from tools.calculator import calculator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from agents.verifier_agent import VerifierAgent

from response import TutorResponse, VerificationResult

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tutor_response: TutorResponse | None

    verification: VerificationResult | None

tutor = TutorAgent()
verifier = VerifierAgent()

def tutor_node(state:AgentState):

    response = tutor.invoke(
        state['messages']
    )

    print("\n[TUTOR]")
    print("Content:", response.content)
    print("Tool calls:", response.tool_calls)

    return {
        "messages": [response]
    }


def verifier_node(state:AgentState):
    user_query = state["messages"][0].content

    tutor_response = state["messages"][-1].content

    result = verifier.verify(
        user_query,
        tutor_response
    )

    print("\n[VERIFIER]")
    print("Valid:", result.valid)
    print("Reason:", result.reason)

    return {
        "verification": result
    }

tool_node = ToolNode([calculator])

def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    # If LLM requested a tool
    if last_message.tool_calls:

        return "tools"

    # Otherwise finish
    return END

builder = StateGraph(AgentState)

builder.add_node(
    "tutor",tutor_node
)

builder.add_node(
    "tools",tool_node
)

builder.add_node(
    "verifier",
    verifier_node
)

builder.add_edge(
    START,
    "tutor"
)

builder.add_conditional_edges(
    "tutor",
    should_continue,
    {
        "tools": "tools",
        END: "verifier"
    }
)

builder.add_edge(
    "tools",
    "tutor"
)

builder.add_edge(
    "verifier",
    END
)

graph = builder.compile()