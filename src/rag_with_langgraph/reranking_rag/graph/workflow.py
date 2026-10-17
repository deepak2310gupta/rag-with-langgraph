from langgraph.graph import StateGraph, START, END

from .state import RerankingRAGState

from .nodes.retrieve_documents import create_retrieve_documents_node
from .nodes.rerank_documents import create_rerank_documents_node
from .nodes.generate_answer import create_generate_answer_node


def build_reranking_rag_graph(
    retriever, reranker, llm, prompt_template, RERANK_TOP_K
):

    graph = StateGraph(RerankingRAGState)

    # Create nodes with their dependencies
    retrieve_documents_node = create_retrieve_documents_node(retriever)
    rerank_documents_node = create_rerank_documents_node(reranker, RERANK_TOP_K)
    generate_answer_node = create_generate_answer_node(prompt_template, llm)

    # Add nodes
    graph.add_node("retrieve_documents", retrieve_documents_node)
    graph.add_node("rerank_documents", rerank_documents_node)
    graph.add_node("generate_answer", generate_answer_node)

    # Add edges
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "rerank_documents")
    graph.add_edge("rerank_documents", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
