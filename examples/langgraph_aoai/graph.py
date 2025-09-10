from __future__ import annotations
from typing import Dict, Any, Tuple
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .models import (
    UserMessage,
    RouteDecision,
    FinalAnswer,
    SummaryResult,
)
from .agents import (
    router_llm,
    answer_llm,
    rag_answer_llm,
    summarizer_llm,
    run_tool_agent,  # new: AgentExecutor hook
)

State = Dict[str, Any]
checkpointer = MemorySaver()

# ---------------------
# Helpers
# ---------------------
def parse_tool_command(text: str) -> str:
    """
    If user types `tool: ...`, strip the prefix and return the remainder for the agent.
    Example inputs:
      tool: echo hello world
      tool: add x=2 y=3
    """
    return text.split(":", 1)[1].strip() if ":" in text else text

# ---------------------
# Nodes
# ---------------------
def router_node(state: State) -> State:
    user: UserMessage = state["user"]
    txt = user.text.strip()
    # Explicit tool mode if user types "tool: ..."
    if txt.lower().startswith("tool:"):
        route = "tool"
    else:
        raw = router_llm(user.text).lower()
        route = "rag" if "rag" in raw else "answer"
    state["decision"] = RouteDecision(route=route)
    return state

# TODO: Replace stub with Azure AI Search retrieval
# 1) pip install azure-search-documents
# 2) set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX
# 3) query SearchClient for top-K docs (id, content, score)
# 4) build `context` and pass to rag_answer_llm(user.text, context)
def rag_node(state: State) -> State:
    user: UserMessage = state["user"]
    context = "[Stubbed RAG results — replace with Azure AI Search]"
    state["rag_result"] = rag_answer_llm(user.text, context)
    return state

def answer_node(state: State) -> State:
    user: UserMessage = state["user"]
    out = answer_llm(user.text)
    state["final"] = FinalAnswer(text=out)
    return state

def tool_node(state: State) -> State:
    user: UserMessage = state["user"]
    agent_input = parse_tool_command(user.text)
    try:
        result = run_tool_agent(agent_input)
    except Exception as e:
        result = f"tool error: {e}"
    state["tool_result"] = {"name": "agent", "result": result}
    return state

def compose_node(state: State) -> State:
    decision: RouteDecision = state["decision"]
    if decision.route == "rag":
        state["final"] = FinalAnswer(text=state["rag_result"])
    elif decision.route == "tool":
        tr = state.get("tool_result", {})
        state["final"] = FinalAnswer(text=f"ToolAgent -> {tr.get('result')}")
    # answer path already set "final"
    return state

def summarize_node(state: State) -> State:
    final: FinalAnswer = state.get("final")
    if final:
        s = summarizer_llm(final.text)
        state["summary"] = SummaryResult(summary=s)
    return state

# ---------------------
# Graph wiring
# ---------------------
graph = StateGraph(State)

graph.add_node("router", router_node)
graph.add_node("rag", rag_node)
graph.add_node("answer", answer_node)
graph.add_node("tool", tool_node)        # now powered by AgentExecutor
graph.add_node("compose", compose_node)
graph.add_node("summarize", summarize_node)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    lambda s: s["decision"].route,
    {"rag": "rag", "answer": "answer", "tool": "tool"},
)
graph.add_edge("rag", "compose")
graph.add_edge("answer", "compose")
graph.add_edge("tool", "compose")
graph.add_edge("compose", "summarize")
graph.add_edge("summarize", END)

app = graph.compile(checkpointer=checkpointer)



