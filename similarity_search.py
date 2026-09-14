import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path

class SimilaritySearchAgent:
    def __init__(self, threshold: float = 0.7):
        """Initialize similarity search agent with threshold."""
        self.threshold = threshold
    
    def compute_similarity_matrix(self, embeddings: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
        """
        Compute pairwise similarity matrix for all documents.
        
        Args:
            embeddings: Dict mapping document IDs to embedding vectors
            
        Returns:
            Tuple of (document_ids, similarity_matrix)
        """
        doc_ids = list(embeddings.keys())
        n_docs = len(doc_ids)
        
        # Create matrix of embeddings
        embedding_matrix = np.array([embeddings[doc_id] for doc_id in doc_ids])
        
        # Normalize embeddings
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        normalized_embeddings = embedding_matrix / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return doc_ids, similarity_matrix
    
    def find_matches_for_query(
        self,
        query_id: str,
        embeddings: Dict[str, np.ndarray],
        threshold: float = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find documents similar to a query document.
        
        Args:
            query_id: ID of query document
            embeddings: Dict of document embeddings
            threshold: Minimum similarity threshold (uses default if None)
            top_k: Number of top results to return
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        if threshold is None:
            threshold = self.threshold
            
        if query_id not in embeddings:
            raise ValueError(f"Query document {query_id} not found in embeddings")
        
        query_embedding = embeddings[query_id]
        similarities = []
        
        for doc_id, doc_embedding in embeddings.items():
            if doc_id == query_id:
                continue  # Skip self
                
            # Compute cosine similarity
            norm1 = query_embedding / np.linalg.norm(query_embedding)
            norm2 = doc_embedding / np.linalg.norm(doc_embedding)
            similarity = float(np.dot(norm1, norm2))
            
            if similarity >= threshold:
                similarities.append((doc_id, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def analyze_distribution(self, similarities: List[float]) -> Dict:
        """Analyze distribution of similarity scores."""
        if not similarities:
            return {}
        
        scores = [score for _, score in similarities]
        
        return {
            "count": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        }
    
    def save_results(self, results: List[Tuple[str, float]], output_path: str):
        """Save results to JSON file."""
        output_data = {
            "matches": [
                {"document": doc_id, "similarity": score}
                for doc_id, score in results
            ],
            "threshold": self.threshold
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_path}")