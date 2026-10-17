from langchain_core.prompts import ChatPromptTemplate


def create_prompt():

    return ChatPromptTemplate.from_template("""
        You are a helpful assistant.

        Answer the question using only the provided context.

        If the answer cannot be found in the context, say:
        "I could not find the answer in the provided documents."

        Context:
        {context}

        Question:
        {question}
        """)
