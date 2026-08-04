import json
import os

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from app.config import settings

_embeddings = None
_vectorstore = None


class LocalFastEmbedEmbeddings(Embeddings):
    """Thin wrapper around fastembed's TextEmbedding, implementing the
    LangChain Embeddings interface directly (bypasses langchain_community's
    FastEmbedEmbeddings wrapper, which silently breaks on newer fastembed
    versions)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()


def get_embeddings():
    """Local, free embedding model (no API key needed) via FastEmbed/ONNX."""
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalFastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings


def build_or_load_vectorstore(faq_path: str = "data/faq.json"):
    """Loads the persisted Chroma collection if it exists, otherwise builds
    it once from data/faq.json and persists it to CHROMA_DIR."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()

    if os.path.exists(settings.CHROMA_DIR) and os.listdir(settings.CHROMA_DIR):
        _vectorstore = Chroma(
            persist_directory=settings.CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="faq",
        )
        return _vectorstore

    with open(faq_path, "r") as f:
        faq_items = json.load(f)

    docs = [
        Document(
            page_content=f"Q: {item['question']}\nA: {item['answer']}",
            metadata={"question": item["question"], "answer": item["answer"]},
        )
        for item in faq_items
    ]

    _vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DIR,
        collection_name="faq",
        collection_metadata={"hnsw:space": "cosine"},
    )
    return _vectorstore


def search_faq(query: str, k: int = 2):
    """Returns [(Document, relevance_score), ...] sorted by relevance.
    Uses raw cosine distance instead of langchain's built-in relevance-score
    helper, which returns unreliable/zeroed scores on some Chroma setups."""
    vs = build_or_load_vectorstore()
    results = vs.similarity_search_with_score(query, k=k)
    return [(doc, 1 - distance) for doc, distance in results]
