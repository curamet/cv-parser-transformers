from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime
from loguru import logger
from config.settings import settings

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant not available. Install with: pip install qdrant-client")

class QdrantEngine:
    """Qdrant vector database engine for high-performance similarity search and storage."""
    
    def __init__(self):
        """Initialize Qdrant engine."""
        self.client = None
        self.collection_name = settings.VECTOR_DB_SETTINGS.get("qdrant_collection_name", "cv-job-matcher")
        self.host = settings.VECTOR_DB_SETTINGS.get("qdrant_host", "localhost")
        self.port = settings.VECTOR_DB_SETTINGS.get("qdrant_port", 6333)
        self.api_key = settings.VECTOR_DB_SETTINGS.get("qdrant_api_key")
        
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
        
        # Initialize Qdrant
        self._initialize_qdrant()
    
    def _initialize_qdrant(self):
        """Initialize Qdrant client and collection."""
        try:
            # Initialize Qdrant client
            if self.api_key:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key
                )
            else:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port
                )
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=768,  # Standard embedding dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Connected to existing Qdrant collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {str(e)}")
            raise
    
    def store_vectors(self, documents: List[Dict]) -> List[str]:
        """
        Store document vectors in Qdrant.
        
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
                
                # Store vectors in Qdrant
                self._store_document_vectors(doc_id, embeddings, raw_text, doc_type)
                stored_ids.append(doc_id)
                
                logger.info(f"Stored vectors in Qdrant for: {doc_id}")
                
            except Exception as e:
                logger.error(f"Error storing vectors in Qdrant for {doc_data.get('doc_id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Successfully stored {len(stored_ids)} document vectors in Qdrant")
        return stored_ids
    
    def _store_document_vectors(self, doc_id: str, embeddings: Dict, raw_text: str, doc_type: str):
        """
        Store document vectors in Qdrant.
        
        Args:
            doc_id: Document ID
            embeddings: Dictionary containing embeddings
            raw_text: Original document text
            doc_type: Document type ("cv" or "job")
        """
        if self.client is None:
            raise Exception("Qdrant client not initialized")
            
        try:
            points = []
            
            # Store full document embedding
            points.append(PointStruct(
                id=f"{doc_id}_full",
                vector=embeddings["full_document"].tolist(),
                payload={
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "section": "full_document",
                    "raw_text": raw_text,
                    "stored_at": datetime.now().isoformat()
                }
            ))
            
            # Store section embeddings
            for section_name, section_embedding in embeddings["sections"].items():
                points.append(PointStruct(
                    id=f"{doc_id}_{section_name}",
                    vector=section_embedding.tolist(),
                    payload={
                        "doc_id": doc_id,
                        "doc_type": doc_type,
                        "section": section_name,
                        "raw_text": raw_text,
                        "stored_at": datetime.now().isoformat()
                    }
                ))
            
            # Upsert points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Stored vectors in Qdrant for document {doc_id} with {len(embeddings['sections'])} sections")
            
        except Exception as e:
            logger.error(f"Error storing vectors in Qdrant for document {doc_id}: {str(e)}")
            raise
    
    def search_similar_documents(self, query_embedding: np.ndarray, 
                               doc_type: str = None, 
                               section: str = None,
                               n_results: int = 10) -> List[Dict]:
        """
        Search for similar documents using Qdrant.
        
        Args:
            query_embedding: Query embedding
            doc_type: Filter by document type ("cv" or "job")
            section: Filter by section name
            n_results: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if self.client is None:
            logger.error("Qdrant client not initialized")
            return []
            
        try:
            # Build filter
            filter_conditions = []
            if doc_type:
                filter_conditions.append({"key": "doc_type", "match": {"value": doc_type}})
            if section:
                filter_conditions.append({"key": "section", "match": {"value": section}})
            
            # Search in Qdrant collection
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=n_results,
                query_filter={"must": filter_conditions} if filter_conditions else None,
                with_payload=True
            )
            
            # Format results
            similar_docs = []
            for result in results:
                doc_info = {
                    "doc_id": result.payload["doc_id"],
                    "doc_type": result.payload["doc_type"],
                    "section": result.payload["section"],
                    "similarity_score": result.score,
                    "stored_at": result.payload["stored_at"]
                }
                similar_docs.append(doc_info)
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching similar documents in Qdrant: {str(e)}")
            return []
    
    def get_document_embeddings(self, doc_id: str, section: str = None) -> Optional[np.ndarray]:
        """
        Retrieve embeddings for a specific document from Qdrant.
        
        Args:
            doc_id: Document ID
            section: Section name (if None, returns full document embedding)
            
        Returns:
            Document embedding or None
        """
        if self.client is None:
            logger.error("Qdrant client not initialized")
            return None
            
        try:
            # Build query ID
            query_id = f"{doc_id}_{section}" if section else f"{doc_id}_full"
            
            # Retrieve from Qdrant
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[query_id],
                with_vectors=True
            )
            
            if results:
                return np.array(results[0].vector)
            else:
                logger.warning(f"No embedding found in Qdrant for {query_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving embedding from Qdrant for {doc_id}: {str(e)}")
            return None
    
    def get_document_sections(self, doc_id: str) -> List[str]:
        """
        Get available sections for a document from Qdrant.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of available section names
        """
        if self.client is None:
            logger.error("Qdrant client not initialized")
            return []
            
        try:
            # Scroll through all points with this doc_id
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
                limit=1000,
                with_payload=True
            )
            
            sections = []
            for point in results[0]:
                sections.append(point.payload["section"])
            
            return list(set(sections))
            
        except Exception as e:
            logger.error(f"Error retrieving sections from Qdrant for {doc_id}: {str(e)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its embeddings from Qdrant.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.error("Qdrant client not initialized")
            return False
            
        try:
            # Get all point IDs for this document
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={"must": [{"key": "doc_id", "match": {"value": doc_id}}]},
                limit=1000,
                with_payload=False
            )
            
            if results[0]:
                point_ids = [point.id for point in results[0]]
                # Delete points
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                logger.info(f"Deleted document {doc_id} with {len(point_ids)} embeddings from Qdrant")
                return True
            else:
                logger.warning(f"No embeddings found in Qdrant for document {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document from Qdrant {doc_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the Qdrant collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        if self.client is None:
            logger.error("Qdrant client not initialized")
            return {}
            
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = {
                "database_type": "qdrant",
                "total_embeddings": collection_info.points_count,
                "cv_count": 0,
                "job_count": 0,
                "sections": {},
                "documents": {}
            }
            
            # Scroll through all points to get detailed stats
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )
            
            for point in results[0]:
                payload = point.payload
                doc_type = payload["doc_type"]
                section = payload["section"]
                doc_id = payload["doc_id"]
                
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
            logger.error(f"Error getting Qdrant collection stats: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the Qdrant collection.
        
        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            logger.error("Qdrant client not initialized")
            return False
            
        try:
            # Delete all points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={"all": True}
            )
            logger.info("Cleared all data from Qdrant collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing Qdrant collection: {str(e)}")
            return False

# Global Qdrant engine instance
qdrant_engine = QdrantEngine() 