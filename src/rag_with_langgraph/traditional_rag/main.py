from dotenv import load_dotenv

from .graph.workflow import build_traditional_rag_graph

from ..common.ingestion.loader import load_documents
from ..common.chunking.splitter import split_documents
from ..common.embeddings.embedding_model import get_embedding_model
from ..common.vectorStore.vector_store import create_vector_store
from ..common.retriever.retriever import create_retriever
from ..common.generation.llm import create_llm
from ..common.generation.prompt import create_prompt

# Run:
# uv run python -m rag_with_langgraph.traditional_rag.main

# The above run command syntax is different from this command syntax
# uv run python src\rag_with_langgraph\traditional_rag\main.py
# because the first command uses the module syntax with -m, which allows Python to locate the module within the package structure,
# while the second command directly specifies the file path.
# If we want to use the second command to run the project then
# we need to change the relative imports to absolute imports.


def main():

    load_dotenv()

    # 1. Load and split documents
    DIRECTORY_PATH = "src/rag_with_langgraph/traditional_rag/data"
    documents = load_documents(DIRECTORY_PATH)
    chunks = split_documents(documents)

    # 2. Get embedding model
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)

    # 3. Create vector store
    vector_store = create_vector_store(chunks, embeddings)

    # 4. Create dependencies
    RETRIEVE_TOP_K = 5
    retriever = create_retriever(vector_store, RETRIEVE_TOP_K)
    prompt_template = create_prompt()
    llm = create_llm()

    # 5. Build graph
    graph = build_traditional_rag_graph(
        retriever=retriever,
        prompt_template=prompt_template,
        llm=llm,
    )

    # 6. Get question
    question = input("Enter your question: ")

    # 7. Invoke graph
    result = graph.invoke({"question": question, "documents": [], "answer": ""})

    # 8. Print answer
    print("\nAnswer:")
    print(result["answer"])


if __name__ == "__main__":
    main()
