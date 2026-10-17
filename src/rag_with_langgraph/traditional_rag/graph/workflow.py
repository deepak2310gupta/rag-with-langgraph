from langgraph.graph import StateGraph, START, END

from .state import RAGState

from .nodes.retrieve_documents import create_retrieve_documents_node
from .nodes.generate_answer import create_generate_answer_node


def build_traditional_rag_graph(retriever, llm, prompt_template):

    graph = StateGraph(RAGState)

    # Create nodes with their dependencies
    retrieve_documents_node = create_retrieve_documents_node(retriever)
    generate_answer_node = create_generate_answer_node(prompt_template, llm)

    # Add nodes
    graph.add_node("retrieve_documents", retrieve_documents_node)
    graph.add_node("generate_answer", generate_answer_node)

    # Add edges
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
