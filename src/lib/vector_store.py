import os
import numpy as np
import faiss
from typing import List, Optional, Dict, Any

class FaissVectorStore:
    def __init__(self, dimension: int):
        """
        Initialize a FAISS vector store
        :param dimension: Dimensionality of the embeddings
        """
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.texts = []
    
    def add_embeddings(
        self, 
        embeddings: List[np.ndarray], 
        texts: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add embeddings to the vector store
        :param embeddings: Numpy array of embeddings
        :param texts: Corresponding texts
        :param metadata: Optional metadata for each embedding
        """
        # Add embeddings to FAISS index
        for embedding in embeddings:
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            self.index.add(embedding)
        # Store texts
        self.texts.extend(texts)
        # Store metadata (use empty dict if not provided)
        metadata = metadata or [{} for _ in range(len(texts))]
        self.metadata.extend(metadata)
    
    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Perform similarity search
        :param query_embedding: Query embedding vector
        :param k: Number of top results to return
        :param filter_metadata: Optional metadata filter
        :return: Tuple of (distances, indices, texts, metadata)
        """
        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        # Perform similarity search
        distances, indices = self.index.search(query_embedding, k)
        
        # # Filter results based on metadata if specified
        # if filter_metadata:
        #     filtered_results = []
        #     for dist, idx in zip(distances[0], indices[0]):
        #         # Check if metadata matches filter
        #         if all(
        #             self.metadata[idx].get(key) == value 
        #             for key, value in filter_metadata.items()
        #         ):
        #             filtered_results.append((dist, idx))
        #     # If no results match, return empty lists
        #     if not filtered_results:
        #         return [], [], [], []
        #     # Unpack filtered results
        #     distances = np.array([res[0] for res in filtered_results]).reshape(1, -1)
        #     indices = np.array([res[1] for res in filtered_results]).reshape(1, -1)
        
        # Retrieve corresponding texts and metadata
        result_texts = [self.texts[idx] for idx in indices[0]]
        result_metadata = [self.metadata[idx] for idx in indices[0]]
        return distances, indices, result_texts, result_metadata
    
    def save(self, directory: str):
        """
        Save the vector store to disk
        :param directory: Directory to save vector store components
        """
        os.makedirs(directory, exist_ok=True)
        # Store
        faiss.write_index(self.index, os.path.join(directory, 'faiss_index.index'))
        np.save(os.path.join(directory, 'texts.npy'), self.texts)
        np.save(os.path.join(directory, 'metadata.npy'), self.metadata)
    
    @classmethod
    def load(cls, directory: str, dimension: int):
        """
        Load a previously saved vector store
        :param directory: Directory containing saved vector store
        :param dimension: Dimensionality of embeddings
        :return: Loaded FaissVectorStore instance
        """
        # Create a new instance
        vector_store = cls(dimension)
        # Load
        vector_store.index = faiss.read_index(os.path.join(directory, 'faiss_index.index'))
        vector_store.texts = np.load(os.path.join(directory, 'texts.npy'), allow_pickle=True).tolist()
        vector_store.metadata = np.load(os.path.join(directory, 'metadata.npy'), allow_pickle=True).tolist()
        return vector_store