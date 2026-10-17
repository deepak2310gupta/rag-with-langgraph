from sentence_transformers import CrossEncoder


def create_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
):
    return CrossEncoder(model_name)