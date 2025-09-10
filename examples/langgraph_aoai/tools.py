from typing import Dict, Any
from langchain.tools import tool

@tool
def echo_tool(text: str) -> Dict[str, Any]:
    """Echo back text (placeholder for real Azure/1P tools)."""
    return {"echo": text}

# TODO: # Integrate: call echo_tool in a new node or inside rag/answer for hybrid flows.