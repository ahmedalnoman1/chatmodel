"""
Arabic Cybercrime RAG Chatbot - Source Package
نظام الدردشة العربي لبلاغات الجرائم الإلكترونية
"""

# Lazy imports to avoid loading all dependencies at once
__all__ = [
    "ArabicProcessor",
    "DocumentLoader", 
    "ArabicEmbeddings",
    "FAISSVectorStore",
    "BM25Retriever",
    "HybridRetriever",
    "ArabicReranker",
    "LMStudioClient",
    "CybercrimeAssistant",
    "PDFReporter"
]

def __getattr__(name):
    """Lazy loading of modules"""
    if name == "ArabicProcessor":
        from .arabic_processor import ArabicProcessor
        return ArabicProcessor
    elif name == "DocumentLoader":
        from .document_loader import DocumentLoader
        return DocumentLoader
    elif name == "ArabicEmbeddings":
        from .embeddings import ArabicEmbeddings
        return ArabicEmbeddings
    elif name == "FAISSVectorStore":
        from .vector_store import FAISSVectorStore
        return FAISSVectorStore
    elif name == "BM25Retriever":
        from .bm25_retriever import BM25Retriever
        return BM25Retriever
    elif name == "HybridRetriever":
        from .hybrid_retriever import HybridRetriever
        return HybridRetriever
    elif name == "ArabicReranker":
        from .reranker import ArabicReranker
        return ArabicReranker
    elif name == "LMStudioClient":
        from .llm_client import LMStudioClient
        return LMStudioClient
    elif name == "CybercrimeAssistant":
        from .chatbot import CybercrimeAssistant
        return CybercrimeAssistant
    elif name == "PDFReporter":
        from .pdf_reporter import PDFReporter
        return PDFReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
