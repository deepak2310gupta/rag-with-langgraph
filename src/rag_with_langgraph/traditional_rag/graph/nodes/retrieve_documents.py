from ..state import RAGState


def create_retrieve_documents_node(retriever):

    def retrieve_documents(state: RAGState):

        question = state["question"]

        print("\n🔍 [Retrieve] Searching documents...")
        print(f"Question: {question}")

        documents = retriever.invoke(question)

        print(f"Retrieved documents: {len(documents)}")

        return {"documents": documents}

    return retrieve_documents
