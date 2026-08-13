import os
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ==========================================
# 1. State Definition
# ==========================================
class AgentState(TypedDict):
    # Maintains full conversation history using the add_messages reducer
    messages: Annotated[list[BaseMessage], add_messages]
    # Tracks execution metadata
    current_step: str
    error_count: int

# ==========================================
# 2. Tool Definitions (2 Dedicated Tools)
# ==========================================
@tool
def legal_code_search(query: str) -> str:
    """Searches Pakistani legal statutes (PPC, PECA, CrPC, Contract Act) for specific legal provisions."""
    try:
        query_lower = query.lower()
        if "cyber" in query_lower or "peca" in query_lower or "defamation" in query_lower:
            return "PECA Section 14: Unauthorized use of identity data / online defamation carries up to 3 years imprisonment or fine."
        elif "bail" in query_lower:
            return "CrPC Section 497: Governs conditions for granting bail in non-bailable offenses under Pakistani criminal law."
        elif "contract" in query_lower or "agreement" in query_lower:
            return "Contract Act 1872 Section 2(h): An agreement enforceable by law is a valid contract."
        else:
            return f"PPC Reference Result: Found general statutory precedent applicable to query: '{query}'."
    except Exception as e:
        return f"TOOL_ERROR: Legal search query failed. Details: {str(e)}"

@tool
def plain_english_explainer(legal_text: str) -> str:
    """Translates complex Pakistani legal citations into clear, everyday language with practical examples."""
    try:
        if not legal_text or "TOOL_ERROR" in legal_text:
            raise ValueError("Invalid statutory input provided for simplification.")
        
        return (
            f"Simplified Legal Context: [{legal_text}]\n"
            "Practical Meaning: In simple terms, this statute lays down clear ground rules. "
            "If someone violates these provisions, the affected person can lodge a complaint with law enforcement "
            "or seek civil remedies in court."
        )
    except Exception as e:
        return f"TOOL_ERROR: Explanation formatting failed. Details: {str(e)}"

tools = [legal_code_search, plain_english_explainer]
tool_node = ToolNode(tools)

# ==========================================
# 3. Model Binding
# ==========================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 4. Core Graph Nodes
# ==========================================
def agent_node(state: AgentState) -> dict:
    """Primary reasoning node evaluating state to decide next action or final response."""
    messages = state["messages"]
    
    system_prompt = SystemMessage(
        content=(
            "You are a specialized Legal AI Assistant for Pakistan Law.\n"
            "Follow this execution path:\n"
            "1. First, search statutory provisions using `legal_code_search`.\n"
            "2. Next, pass the raw legal findings into `plain_english_explainer` to generate an accessible summary.\n"
            "3. Finally, combine both outputs into a clear, structured final answer for the user."
        )
    )
    
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {
        "messages": [response],
        "current_step": "agent_reasoning",
        "error_count": state.get("error_count", 0)
    }

def error_recovery_node(state: AgentState) -> dict:
    """Graceful error recovery node invoked when tool execution yields an explicit failure."""
    print("⚠️ Tool execution issue detected. Executing graceful fallback...")
    recovery_msg = AIMessage(
        content="I encountered a temporary issue looking up the exact statute code, but I can still explain the general framework under Pakistani law based on standard legal procedures."
    )
    return {
        "messages": [recovery_msg],
        "current_step": "error_handled",
        "error_count": state.get("error_count", 0) + 1
    }

# ==========================================
# 5. Router Logic (Conditional Routing)
# ==========================================
def should_continue(state: AgentState) -> Literal["tools", "error_recovery", "__end__"]:
    """Routes execution based on whether the agent made tool calls or if a tool returned an error."""
    messages = state["messages"]
    last_message = messages[-1]

    # Intercept tool error responses
    if isinstance(last_message, ToolMessage) and "TOOL_ERROR" in str(last_message.content):
        return "error_recovery"

    # Route to ToolNode if LLM generated tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END

# ==========================================
# 6. Graph Construction
# ==========================================
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("error_recovery", error_recovery_node)

# Flow Connections
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "error_recovery": "error_recovery",
        END: END
    }
)

# Route tool outputs back to agent for synthesis
workflow.add_edge("tools", "agent")
workflow.add_edge("error_recovery", "agent")

app = workflow.compile()

# ==========================================
# 7. Execution & Trace
# ==========================================
if __name__ == "__main__":
    # Visual ASCII Graph Output
    print("=== LangGraph Execution Visualizer ===")
    try:
        app.get_graph().print_ascii()
    except Exception:
        pass

    initial_input: AgentState = {
        "messages": [
            HumanMessage(content="What are the penalties for cyber defamation under PECA in Pakistan, and what does it mean for an average citizen?")
        ],
        "current_step": "init",
        "error_count": 0
    }

    print("\n=== Running Multi-Step Workflow ===\n")
    for event in app.stream(initial_input, stream_mode="values"):
        latest = event["messages"][-1]
        sender = latest.__class__.__name__
        
        print(f"[{sender}]")
        if hasattr(latest, "tool_calls") and latest.tool_calls:
            print(f"🛠️ Selected Tool Call: {latest.tool_calls}\n")
        else:
            print(f"{latest.content}\n" + "-" * 50)
