Simplified LangGraph + Azure OpenAI Demo
----------------------------------------
* Multi-agent style workflow (router → rag/answer → compose)
* Pydantic v2.10+ models for structure
* Azure OpenAI integration via LangChain
* Stubbed RAG node (can be extended to Azure AI Search)
* There's a seam for MCP tool orchestration

How to run:
1. Set environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT)
2. pip install -r requirements.txt
3. python -m examples.langgraph_aoai.run
