from ..state import RerankingRAGState


def create_rerank_documents_node(reranker, RERANK_TOP_K):

    def rerank_documents(state: RerankingRAGState):

        question = state["question"]
        documents = state["documents"]

        print("\n📊 [Re-Rank] Re-ranking documents...")
        print(f"Documents to re-rank: {len(documents)}")

        # Create query-document pairs
        pairs = [(question, document.page_content) for document in documents]

        # Get relevance scores
        scores = reranker.predict(pairs)

        # Combine documents and scores
        scored_documents = list(zip(documents, scores))

        # Sort by relevance score
        scored_documents.sort(key=lambda item: item[1], reverse=True)

        print("\n" + "-" * 70)
        print("RE-RANKED RESULTS")
        print("-" * 70)

        for index, (document, score) in enumerate(scored_documents, start=1):
            content = document.page_content.strip()

            # Print only first 300 characters
            preview = content[:300]

            if len(content) > 300:
                preview += "..."

            print(f"\nRank {index}")
            print(f"Score: {score:.4f}")
            print(f"Content: {preview}")
            print("-" * 70)

        # Keep top K documents
        reranked_documents = [
            document for document, score in scored_documents[:RERANK_TOP_K]
        ]

        print(f"\nSelected top {len(reranked_documents)} documents")

        return {"reranked_documents": reranked_documents}

    return rerank_documents
