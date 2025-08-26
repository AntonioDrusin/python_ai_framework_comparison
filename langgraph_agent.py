import re
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Tools
@tool
def calculator(expr: str) -> str:
    """ Evaluates the expression in python and returns the result"""
    return str(eval(expr))

tools = [calculator]


# LLM
llm = ChatOpenAI(model="qwen/qwen3-4b", base_url="http://localhost:1234/v1", api_key="not_required")
llm_with_tools = llm.bind_tools(tools)


# State
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Nodes   
def call_model(state: AgentState) -> AgentState:
    print("STATE IN CALL MODEL: ", state)
    messages = state["messages"]

    sysmsg = SystemMessage(
        content=(
            "TOOLS ARE AUTHORITATIVE. If you call a tool and it returns a result, "
            "do NOT verify, recompute, or override it. Return the tool's result directly "
            "(with minimal formatting). Only add reasoning if the user explicitly asks. "
            "If no tool is used, answer normally."
        )
    )
    
    response = llm_with_tools.invoke([sysmsg, *messages])
    state["messages"] = response
    return state

def should_continue(state: AgentState) -> str:
    print("STATE SHOULD CONTINUE: ", state)
    messages = state["messages"]
    last_message = messages[-1]
    print("LAST MESSAGE: ", last_message)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

graph = workflow.compile()

# Initialize the state
initial_state = {
    "messages": [HumanMessage(content="What is the result of 123*456 but talk like a pirate")]
}

result = graph.invoke(initial_state)

#Print the think part and the result part separately
last = result["messages"][-1]
content = last.content if isinstance(last.content, str) else str(last.content)
rest_part = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

print("RESPONSE")
print(rest_part)