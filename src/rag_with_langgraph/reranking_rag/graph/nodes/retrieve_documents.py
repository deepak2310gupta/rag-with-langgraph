from ..state import RerankingRAGState


def create_retrieve_documents_node(retriever):

    def retrieve_documents(state: RerankingRAGState):

        question = state["question"]

        print("\n🔍 [Retrieve] Searching documents...")
        print(f"Question: {question}")

        documents = retriever.invoke(question)

        print(f"Retrieved documents: {len(documents)}")

        for i, doc in enumerate(documents, start=1):
            print("\n" + "=" * 70)
            print(f"DOCUMENT {i}")
            print("=" * 70)

            print("\n[Content]")
            print(doc.page_content)

        print("-" * 70)

        return {"documents": documents}

    return retrieve_documents
