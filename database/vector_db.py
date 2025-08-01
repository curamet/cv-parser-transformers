from typing import List, Dict, Optional, Any
import numpy as np
from loguru import logger
from config.settings import settings

class VectorDatabase:
    """High-level interface for vector database operations with engine selection."""
    
    def __init__(self):
        """Initialize vector database interface with appropriate engine."""
        self.engine = self._get_engine()
    
    def _get_engine(self):
        """Get the appropriate engine based on configuration."""
        db_type = settings.VECTOR_DB_SETTINGS.get("database_type", "chromadb").lower()
        
        try:
            if db_type == "chromadb":
                from .engines.chroma_engine import chroma_engine
                logger.info("Using ChromaDB engine")
                return chroma_engine
            elif db_type == "pinecone":
                from .engines.pinecone_engine import pinecone_engine
                logger.info("Using Pinecone engine")
                return pinecone_engine
            elif db_type == "qdrant":
                from .engines.qdrant_engine import qdrant_engine
                logger.info("Using Qdrant engine")
                return qdrant_engine
            else:
                logger.warning(f"Unknown database type: {db_type}. Falling back to ChromaDB")
                from .engines.chroma_engine import chroma_engine
                return chroma_engine
        except ImportError as e:
            logger.error(f"Failed to import {db_type} engine: {str(e)}")
            logger.info("Falling back to ChromaDB engine")
            from .engines.chroma_engine import chroma_engine
            return chroma_engine
    
    def store_documents(self, documents: List[Dict]) -> List[str]:
        """
        Store documents in the vector database.
        
        Args:
            documents: List of processed documents with embeddings
            
        Returns:
            List of stored document IDs
        """
        try:
            return self.engine.store_vectors(documents)
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            return []
    
    def search_similar(self, query_text: str, doc_type: str = None, 
                      n_results: int = 10) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_text: Query text
            doc_type: Filter by document type
            n_results: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            from services.nlp_processor import nlp_processor
            
            # Generate query embedding
            query_embedding = nlp_processor.generate_embeddings(query_text)[0]
            
            # Search in vector store
            results = self.engine.search_similar_documents(
                query_embedding=query_embedding,
                doc_type=doc_type,
                n_results=n_results
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    def search_similar_ids(self, query_text: str, doc_type: str = None, 
                          n_results: int = 10) -> List[str]:
        """
        Search for similar documents and return only their IDs.
        
        Args:
            query_text: Query text
            doc_type: Filter by document type
            n_results: Number of results to return
            
        Returns:
            List of document IDs
        """
        try:
            from services.nlp_processor import nlp_processor
            
            # Generate query embedding
            query_embedding = nlp_processor.generate_embeddings(query_text)[0]
            
            # Search in vector store
            results = self.engine.search_similar_documents(
                query_embedding=query_embedding,
                doc_type=doc_type,
                n_results=n_results
            )
            
            # Extract only the document IDs
            doc_ids = []
            for result in results:
                if "doc_id" in result:
                    doc_ids.append(result["doc_id"])
            
            return doc_ids
        except Exception as e:
            logger.error(f"Error searching similar document IDs: {str(e)}")
            return []
    
    def get_document(self, doc_id: str, section: str = None) -> Optional[np.ndarray]:
        """
        Get document embeddings.
        
        Args:
            doc_id: Document ID
            section: Section name (optional)
            
        Returns:
            Document embedding or None
        """
        try:
            return self.engine.get_document_embeddings(doc_id, section)
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {str(e)}")
            return None
    
    def get_document_sections(self, doc_id: str) -> List[str]:
        """
        Get available sections for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of available sections
        """
        try:
            return self.engine.get_document_sections(doc_id)
        except Exception as e:
            logger.error(f"Error getting sections for {doc_id}: {str(e)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector database.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.engine.delete_document(doc_id)
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            return self.engine.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting database statistics: {str(e)}")
            return {}
    
    def clear_database(self) -> bool:
        """
        Clear all data from the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.engine.clear_collection()
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return False

    def get_raw_text(self, doc_id: str) -> Optional[str]:
        """
        Get raw text content of a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Raw text content or None if not found
        """
        try:
            return self.engine.get_raw_text(doc_id)
        except Exception as e:
            logger.error(f"Error getting raw text for document {doc_id}: {str(e)}")
            return None
    
    def get_all_documents(self, doc_type: str = None) -> List[Dict]:
        """
        Get all documents from the vector database.
        
        Args:
            doc_type: Filter by document type (cv, job, or None for all)
            
        Returns:
            List of all documents with their metadata
        """
        try:
            return self.engine.get_all_documents(doc_type)
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector database.
        
        Returns:
            Health check results
        """
        try:
            stats = self.get_statistics()
            
            health_status = {
                "status": "healthy",
                "database_type": stats.get("database_type", "unknown"),
                "total_embeddings": stats.get("total_embeddings", 0),
                "cv_count": stats.get("cv_count", 0),
                "job_count": stats.get("job_count", 0),
                "unique_documents": stats.get("unique_documents", 0)
            }
            
            # Check if engine is accessible
            if self.engine is None:
                health_status["status"] = "unhealthy"
                health_status["error"] = "Engine not accessible"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Global vector database instance
vector_db = VectorDatabase() 