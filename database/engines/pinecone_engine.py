from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime
from loguru import logger
from config.settings import settings

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available. Install with: pip install pinecone-client")

class PineconeEngine:
    """Pinecone vector database engine for cloud-based similarity search and storage."""
    
    def __init__(self):
        """Initialize Pinecone engine."""
        self.index = None
        self.index_name = settings.VECTOR_DB_SETTINGS.get("pinecone_index_name", "cv-job-matcher")
        self.api_key = settings.VECTOR_DB_SETTINGS.get("pinecone_api_key")
        self.environment = settings.VECTOR_DB_SETTINGS.get("pinecone_environment", "us-west1-gcp")
        
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available. Install with: pip install pinecone-client")
        
        if not self.api_key:
            raise ValueError("Pinecone API key not configured in settings")
        
        # Initialize Pinecone
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            # Initialize Pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                # Create new index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=768,  # Standard embedding dimension
                    metric="cosine"
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def store_vectors(self, documents: List[Dict]) -> List[str]:
        """
        Store document vectors in Pinecone.
        
        Args:
            documents: List of processed documents with embeddings
            
        Returns:
            List of stored document IDs
        """
        stored_ids = []
        
        for doc_data in documents:
            try:
                doc_id = doc_data["doc_id"]
                raw_text = doc_data["raw_text"]
                doc_type = doc_data["doc_type"]
                embeddings = doc_data["embeddings"]
                
                # Store vectors in Pinecone
                self._store_document_vectors(doc_id, embeddings, raw_text, doc_type)
                stored_ids.append(doc_id)
                
                logger.info(f"Stored vectors in Pinecone for: {doc_id}")
                
            except Exception as e:
                logger.error(f"Error storing vectors in Pinecone for {doc_data.get('doc_id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Successfully stored {len(stored_ids)} document vectors in Pinecone")
        return stored_ids
    
    def _store_document_vectors(self, doc_id: str, embeddings: Dict, raw_text: str, doc_type: str):
        """
        Store document vectors in Pinecone.
        
        Args:
            doc_id: Document ID
            embeddings: Dictionary containing embeddings
            raw_text: Original document text
            doc_type: Document type ("cv" or "job")
        """
        if self.index is None:
            raise Exception("Pinecone index not initialized")
            
        try:
            vectors_to_upsert = []
            
            # Store full document embedding
            vectors_to_upsert.append({
                "id": f"{doc_id}_full",
                "values": embeddings["full_document"].tolist(),
                "metadata": {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "section": "full_document",
                    "raw_text": raw_text,
                    "stored_at": datetime.now().isoformat()
                }
            })
            
            # Store section embeddings
            for section_name, section_embedding in embeddings["sections"].items():
                vectors_to_upsert.append({
                    "id": f"{doc_id}_{section_name}",
                    "values": section_embedding.tolist(),
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_type": doc_type,
                        "section": section_name,
                        "raw_text": raw_text,
                        "stored_at": datetime.now().isoformat()
                    }
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Stored vectors in Pinecone for document {doc_id} with {len(embeddings['sections'])} sections")
            
        except Exception as e:
            logger.error(f"Error storing vectors in Pinecone for document {doc_id}: {str(e)}")
            raise
    
    def search_similar_documents(self, query_embedding: np.ndarray, 
                               doc_type: str = None, 
                               section: str = None,
                               n_results: int = 10) -> List[Dict]:
        """
        Search for similar documents using Pinecone.
        
        Args:
            query_embedding: Query embedding
            doc_type: Filter by document type ("cv" or "job")
            section: Filter by section name
            n_results: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if self.index is None:
            logger.error("Pinecone index not initialized")
            return []
            
        try:
            # Build metadata filter
            filter_dict = {}
            if doc_type:
                filter_dict["doc_type"] = doc_type
            if section:
                filter_dict["section"] = section
            
            # Search in Pinecone index
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=n_results,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            # Format results
            similar_docs = []
            if results and "matches" in results:
                for match in results["matches"]:
                    doc_info = {
                        "doc_id": match["metadata"]["doc_id"],
                        "doc_type": match["metadata"]["doc_type"],
                        "section": match["metadata"]["section"],
                        "similarity_score": match["score"],
                        "stored_at": match["metadata"]["stored_at"]
                    }
                    similar_docs.append(doc_info)
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching similar documents in Pinecone: {str(e)}")
            return []
    
    def get_document_embeddings(self, doc_id: str, section: str = None) -> Optional[np.ndarray]:
        """
        Retrieve embeddings for a specific document from Pinecone.
        
        Args:
            doc_id: Document ID
            section: Section name (if None, returns full document embedding)
            
        Returns:
            Document embedding or None
        """
        if self.index is None:
            logger.error("Pinecone index not initialized")
            return None
            
        try:
            # Build query ID
            query_id = f"{doc_id}_{section}" if section else f"{doc_id}_full"
            
            # Fetch from Pinecone index
            results = self.index.fetch(ids=[query_id])
            
            if results and "vectors" in results and query_id in results["vectors"]:
                return np.array(results["vectors"][query_id]["values"])
            else:
                logger.warning(f"No embedding found in Pinecone for {query_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving embedding from Pinecone for {doc_id}: {str(e)}")
            return None
    
    def get_document_sections(self, doc_id: str) -> List[str]:
        """
        Get available sections for a document from Pinecone.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of available section names
        """
        if self.index is None:
            logger.error("Pinecone index not initialized")
            return []
            
        try:
            # Query for all vectors with this doc_id
            results = self.index.query(
                vector=[0] * 768,  # Dummy vector
                top_k=1000,
                filter={"doc_id": doc_id},
                include_metadata=True
            )
            
            sections = []
            if results and "matches" in results:
                for match in results["matches"]:
                    sections.append(match["metadata"]["section"])
            
            return list(set(sections))
            
        except Exception as e:
            logger.error(f"Error retrieving sections from Pinecone for {doc_id}: {str(e)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its embeddings from Pinecone.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            logger.error("Pinecone index not initialized")
            return False
            
        try:
            # Get all vector IDs for this document
            results = self.index.query(
                vector=[0] * 768,  # Dummy vector
                top_k=1000,
                filter={"doc_id": doc_id},
                include_metadata=False
            )
            
            if results and "matches" in results:
                vector_ids = [match["id"] for match in results["matches"]]
                # Delete vectors
                self.index.delete(ids=vector_ids)
                logger.info(f"Deleted document {doc_id} with {len(vector_ids)} embeddings from Pinecone")
                return True
            else:
                logger.warning(f"No embeddings found in Pinecone for document {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document from Pinecone {doc_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dictionary containing index statistics
        """
        if self.index is None:
            logger.error("Pinecone index not initialized")
            return {}
            
        try:
            # Get index stats
            index_stats = self.index.describe_index_stats()
            
            stats = {
                "database_type": "pinecone",
                "total_embeddings": index_stats.get("total_vector_count", 0),
                "cv_count": 0,
                "job_count": 0,
                "sections": {},
                "documents": {}
            }
            
            # Query for metadata to get detailed stats
            results = self.index.query(
                vector=[0] * 768,  # Dummy vector
                top_k=1000,
                include_metadata=True
            )
            
            if results and "matches" in results:
                for match in results["matches"]:
                    metadata = match["metadata"]
                    doc_type = metadata["doc_type"]
                    section = metadata["section"]
                    doc_id = metadata["doc_id"]
                    
                    if doc_type == "cv":
                        stats["cv_count"] += 1
                    elif doc_type == "job":
                        stats["job_count"] += 1
                    
                    # Count sections
                    if section not in stats["sections"]:
                        stats["sections"][section] = 0
                    stats["sections"][section] += 1
                    
                    # Count unique documents
                    if doc_id not in stats["documents"]:
                        stats["documents"][doc_id] = {
                            "type": doc_type,
                            "sections": []
                        }
                    if section not in stats["documents"][doc_id]["sections"]:
                        stats["documents"][doc_id]["sections"].append(section)
            
            stats["unique_documents"] = len(stats["documents"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting Pinecone index stats: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the Pinecone index.
        
        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            logger.error("Pinecone index not initialized")
            return False
            
        try:
            # Delete all vectors
            self.index.delete(delete_all=True)
            logger.info("Cleared all data from Pinecone index")
            return True
        except Exception as e:
            logger.error(f"Error clearing Pinecone index: {str(e)}")
            return False

# Global Pinecone engine instance
pinecone_engine = PineconeEngine() 