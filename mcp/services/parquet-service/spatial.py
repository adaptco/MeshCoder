import numpy as np
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger("parquet-service.spatial")

class SpatialIngester:
    def __init__(self, target_dim: int = 3):
        self.target_dim = target_dim
        self.pca = PCA(n_components=target_dim)

    def map_to_3d(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Map high-dimensional embeddings (e.g., 384d or 1536d) to 3D tensors.
        If only one embedding is provided, it returns a random but deterministic projection 
        based on the embedding's own variance (using simple normalization if PCA isn't fit).
        """
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        n_samples = embeddings.shape[0]
        
        # PCA requires at least n_components samples to fit properly
        # As a fallback for fewer samples, we use a projection matrix
        if n_samples < self.target_dim:
            logger.info("Fewer samples than target dimensions; using fixed projection.")
            # Deterministic projection based on first few dimensions
            # In a real model, this would be a pre-trained linear layer
            projection = embeddings[:, :self.target_dim]
            # Normalize to [-1, 1] range for CAD environments
            norm = np.linalg.norm(projection, axis=1, keepdims=True)
            return projection / (norm + 1e-6)
            
        try:
            reduced = self.pca.fit_transform(embeddings)
            # Normalize to [-10, 10] range (standard Blender metric scale)
            min_val = np.min(reduced)
            max_val = np.max(reduced)
            if max_val > min_val:
                reduced = 20 * (reduced - min_val) / (max_val - min_val) - 10
            return reduced
        except Exception as e:
            logger.error(f"Error in PCA transformation: {e}")
            return embeddings[:, :self.target_dim]

def format_for_blender(tensors: np.ndarray) -> list[dict]:
    """
    Format 3D tensors into a Blender-friendly structure for keyframing/mesh deforming.
    """
    results = []
    for tensor in tensors:
        results.append({
            "location": {
                "x": float(tensor[0]),
                "y": float(tensor[1]),
                "z": float(tensor[2])
            },
            "scale": {
                "x": 1.0, "y": 1.0, "z": 1.0
            }
        })
    return results

_ingester = None

def get_spatial_ingester():
    global _ingester
    if _ingester is None:
        _ingester = SpatialIngester()
    return _ingester
