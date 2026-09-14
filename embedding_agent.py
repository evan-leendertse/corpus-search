from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple
import torch

class EmbeddingAgent:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """Initialize embedding agent with specified model."""
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def generate_embeddings(self, texts: Dict[str, str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a dictionary of texts.
        
        Args:
            texts: Dict mapping document IDs to text content
            batch_size: Batch size for embedding generation
            
        Returns:
            Dict mapping document IDs to embedding vectors
        """
        embeddings = {}
        
        # Convert texts to lists for batch processing
        doc_ids = list(texts.keys())
        text_list = list(texts.values())
        
        # Generate embeddings in batches
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            batch_ids = doc_ids[i:i+batch_size]
            
            # Generate embeddings
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=self.device
            )
            
            # Store embeddings
            for doc_id, embedding in zip(batch_ids, batch_embeddings):
                embeddings[doc_id] = embedding
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(norm1, norm2)
        return float(similarity)
    
    def find_similar_documents(
        self, 
        query_embedding: np.ndarray, 
        document_embeddings: Dict[str, np.ndarray],
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find documents similar to query based on embedding similarity.
        
        Args:
            query_embedding: Embedding of query document
            document_embeddings: Dict of document embeddings
            threshold: Minimum similarity threshold
            top_k: Number of top results to return
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        similarities = []
        
        for doc_id, doc_embedding in document_embeddings.items():
            similarity = self.compute_similarity(query_embedding, doc_embedding)
            if similarity >= threshold:
                similarities.append((doc_id, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]