from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    """
    Interface for providing dense vector embeddings.
    Reserved for future Semantic Search, Hybrid Search, and RAG capabilities in Baseline v3.0+.
    """
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Convert a search query into a dense vector."""
        pass
        
    @abstractmethod
    def embed_document(self, text: str) -> List[float]:
        """Convert a document into a dense vector."""
        pass
