from typing import TypedDict
from langchain_core.documents import Document

class RerankingRAGState(TypedDict):
    question: str
    documents: list[Document]
    reranked_documents: list[Document]
    answer: str
