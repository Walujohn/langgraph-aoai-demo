from __future__ import annotations

import time
import backoff
from typing import List, Dict, Any, Optional

from langchain_openai import AzureOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

from langchain.tools import tool  # simple tools for the tool/agent branch

from .azure_config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)

# --- Optional: MLflow (experiment tracking) ---
try:
    import mlflow  # type: ignore
    _MLFLOW = True
except Exception:
    _MLFLOW = False

def make_llm() -> AzureOpenAI:
    """
    Create a LangChain LLM client for Azure OpenAI.

    Demonstrates:
      - LangChain integration (langchain_openai.AzureOpenAI)
      - Azure-specific config (endpoint, deployment, api_version, api_key)
    """
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        temperature=0.2,
        timeout=60,
    )

def _mlflow_log(span: str, prompt: str, response: str, t_ms: float, extra: Optional[Dict[str, Any]] = None) -> None:
    """Minimal MLflow logging helper. No-ops if MLflow isn't installed."""
    if not _MLFLOW:
        return
    with mlflow.start_run(run_name=span, nested=True):
        mlflow.log_param("deployment", AZURE_OPENAI_DEPLOYMENT or "unset")
        mlflow.log_param("api_version", AZURE_OPENAI_API_VERSION or "unset")
        if extra:
            for k, v in extra.items():
                if isinstance(v, (str, int, float, bool)):
                    mlflow.log_param(k, v)
        mlflow.log_metric("latency_ms", t_ms)
        mlflow.log_metric("prompt_chars", len(prompt or ""))
        mlflow.log_metric("response_chars", len(response or ""))

@backoff.on_exception(backoff.expo, Exception, max_time=30)
def call_llm(messages: List, span: str = "llm.call") -> str:
    """Resilient AOAI call via LangChain, with timing + MLflow logging."""
    llm = make_llm()
    t0 = time.perf_counter()
    resp = llm.invoke(messages)
    dt_ms = (time.perf_counter() - t0) * 1000

    prompt_text = ""
    try:
        prompt_text = "\n\n".join([m.content for m in messages if isinstance(m, HumanMessage)])
    except Exception:
        pass

    out = resp.content if isinstance(resp, AIMessage) else str(resp)
    _mlflow_log(span=span, prompt=prompt_text, response=out, t_ms=dt_ms)
    return out

# ---------------------------
# Agents (wrappers)
# ---------------------------

def router_llm(user_text: str) -> str:
    """Classify into 'rag' or 'answer' (tool route is explicit with 'tool:' prefix)."""
    prompt = (
        "Classify the user's need:\n"
        "- 'rag' if the question is factual and should consult a knowledge base\n"
        "- 'answer' if a direct short answer is fine\n"
        "Reply with ONLY one word: rag or answer.\n\n"
        f"User: {user_text}"
    )
    return call_llm([HumanMessage(content=prompt)], span="router")


def answer_llm(user_text: str) -> str:
    """Direct answer agent."""
    prompt = f"Answer clearly and concisely:\n\n{user_text}"
    return call_llm([HumanMessage(content=prompt)], span="answer")

# --- RAG with a LangChain PromptTemplate ---
_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Use ONLY the provided context to answer. Provide short citations like [doc 1], [doc 2] when relevant."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def rag_answer_llm(user_text: str, context: str) -> str:
    """RAG answer agent using provided context with a ChatPromptTemplate."""
    messages = _rag_prompt.format_messages(context=context, question=user_text)
    return call_llm(messages, span="rag")


def summarizer_llm(text: str) -> str:
    """Summarize into two sentences."""
    prompt = f"Summarize in two sentences:\n\n{text}"
    return call_llm([HumanMessage(content=prompt)], span="summarize")

# ---------------------------
# Simple LangChain tools
# ---------------------------

@tool
def echo_tool(text: str) -> Dict[str, Any]:
    """Echo text back (placeholder for real tools or MCP-backed tools)."""
    return {"echo": text}

@tool
def add_tool(x: int, y: int) -> int:
    """Add two integers."""
    return x + y

# ---------------------------
# LangChain AgentExecutor using our tools
# ---------------------------

# Small ReAct-style prompt for the tool agent
_tool_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools when needed. Think step-by-step."),
    ("human", "{input}")
])

def run_tool_agent(user_text: str) -> str:
    """Run a tiny LangChain AgentExecutor over echo/add tools."""
    llm = make_llm()
    tools = [echo_tool, add_tool]
    agent = create_react_agent(llm, tools, _tool_agent_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    out = executor.invoke({"input": user_text})
    return out.get("output", str(out))



