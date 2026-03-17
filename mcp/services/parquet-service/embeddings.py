from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger("parquet-service.embeddings")

class EmbeddingKernel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate high-quality vector embeddings for a list of strings."""
        return self.model.encode(texts)

    def compute_similarity(self, query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query and a set of document vectors."""
        # query_vector should be 2D (1, dim), doc_vectors should be 2D (n, dim)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        return cosine_similarity(query_vector, doc_vectors)[0]

# Singleton instance to avoid re-loading model per request
_kernel = None

def get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = EmbeddingKernel()
    return _kernel
