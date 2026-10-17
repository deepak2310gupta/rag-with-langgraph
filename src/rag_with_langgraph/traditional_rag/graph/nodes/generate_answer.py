from ..state import RAGState


def create_generate_answer_node(prompt_template, llm):

    def generate_answer(state: RAGState):

        question = state["question"]
        documents = state["documents"]

        # Build context from reranked documents
        context = "\n\n".join(document.page_content for document in documents)

        prompt = prompt_template.invoke({"context": context, "question": question})

        response = llm.invoke(prompt)

        print("Answer generated successfully.")

        return {"answer": response.content}

    return generate_answer
