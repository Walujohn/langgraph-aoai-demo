# TODO: search_azure.py — real Azure AI Search retrieval (plug into rag_node later)

"""
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

class AzureSearch:
    def __init__(self):
        self.client = SearchClient(
            endpoint=SEARCH_ENDPOINT,
            index_name=SEARCH_INDEX,
            credential=AzureKeyCredential(SEARCH_API_KEY),
        )

    def topk(self, query: str, k: int = 5):
        results = self.client.search(query, top=k)
        out = []
        for r in results:
            out.append({
                "id": str(r.get("id", "")),
                "content": r.get("content", ""),
                "score": getattr(r, "@search.score", 0.0),
                "meta": {k: v for k, v in r.items() if k not in {"id", "content"}},
            })
        return out
"""