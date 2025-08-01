import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime
from loguru import logger
from config.settings import settings

class ChromaEngine:
    """ChromaDB vector database engine for efficient similarity search and storage."""
    
    def __init__(self):
        """Initialize ChromaDB engine."""
        self.client = None
        self.collection = None
        self.collection_name = settings.VECTOR_DB_SETTINGS["chromadb_collection_name"]
        self.persist_directory = settings.VECTOR_DB_SETTINGS["chromadb_persist_directory"]
        
        # Initialize ChromaDB
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=None  # We'll handle embeddings ourselves
                )
                logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "CV and Job embeddings for semantic matching",
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def store_vectors(self, documents: List[Dict]) -> List[str]:
        """
        Store document vectors in ChromaDB.
        
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
                
                # Store vectors in ChromaDB
                self._store_document_vectors(doc_id, embeddings, raw_text, doc_type)
                stored_ids.append(doc_id)
                
                logger.info(f"Stored vectors in ChromaDB for: {doc_id}")
                
            except Exception as e:
                logger.error(f"Error storing vectors in ChromaDB for {doc_data.get('doc_id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Successfully stored {len(stored_ids)} document vectors in ChromaDB")
        return stored_ids
    
    def _store_document_vectors(self, doc_id: str, embeddings: Dict, raw_text: str, doc_type: str):
        """
        Store document vectors in ChromaDB.
        
        Args:
            doc_id: Document ID
            embeddings: Dictionary containing embeddings
            raw_text: Original document text
            doc_type: Document type ("cv" or "job")
        """
        if self.collection is None:
            raise Exception("ChromaDB collection not initialized")
            
        try:
            # Store full document embedding
            self.collection.add(
                embeddings=[embeddings["full_document"].tolist()],
                documents=[raw_text],
                metadatas=[{
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "section": "full_document",
                    "stored_at": datetime.now().isoformat()
                }],
                ids=[f"{doc_id}_full"]
            )
            
            # Store section embeddings
            for section_name, section_embedding in embeddings["sections"].items():
                self.collection.add(
                    embeddings=[section_embedding.tolist()],
                    documents=[raw_text],  # Store full text for each section
                    metadatas=[{
                        "doc_id": doc_id,
                        "doc_type": doc_type,
                        "section": section_name,
                        "stored_at": datetime.now().isoformat()
                    }],
                    ids=[f"{doc_id}_{section_name}"]
                )
            
            logger.info(f"Stored vectors in ChromaDB for document {doc_id} with {len(embeddings['sections'])} sections")
            
        except Exception as e:
            logger.error(f"Error storing vectors in ChromaDB for document {doc_id}: {str(e)}")
            raise
    
    def search_similar_documents(self, query_embedding: np.ndarray, 
                               doc_type: str = None, 
                               section: str = None,
                               n_results: int = 10) -> List[Dict]:
        """
        Search for similar documents using ChromaDB.
        
        Args:
            query_embedding: Query embedding
            doc_type: Filter by document type ("cv" or "job")
            section: Filter by section name
            n_results: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return []
            
        try:
            # Build metadata filter
            where_filter = {}
            if doc_type:
                where_filter["doc_type"] = doc_type
            if section:
                where_filter["section"] = section
            
            # Search in ChromaDB collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_filter if where_filter else None,
                include=["metadatas", "distances"]
            )
            
            # Format results
            similar_docs = []
            if results and "ids" in results and results["ids"]:
                for i in range(len(results["ids"][0])):
                    doc_info = {
                        "doc_id": results["metadatas"][0][i]["doc_id"],
                        "doc_type": results["metadatas"][0][i]["doc_type"],
                        "section": results["metadatas"][0][i]["section"],
                        "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                        "stored_at": results["metadatas"][0][i]["stored_at"]
                    }
                    similar_docs.append(doc_info)
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error searching similar documents in ChromaDB: {str(e)}")
            return []
    
    def get_document_embeddings(self, doc_id: str, section: str = None) -> Optional[np.ndarray]:
        """
        Retrieve embeddings for a specific document from ChromaDB.
        
        Args:
            doc_id: Document ID
            section: Section name (if None, returns full document embedding)
            
        Returns:
            Document embedding or None
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return None
            
        try:
            # Build query ID
            query_id = f"{doc_id}_{section}" if section else f"{doc_id}_full"
            
            # Get embedding from ChromaDB collection
            results = self.collection.get(
                ids=[query_id],
                include=["embeddings"]
            )
            
            if results and "embeddings" in results and results["embeddings"]:
                return np.array(results["embeddings"][0])
            else:
                logger.warning(f"No embedding found in ChromaDB for {query_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving embedding from ChromaDB for {doc_id}: {str(e)}")
            return None
    
    def get_document_sections(self, doc_id: str) -> List[str]:
        """
        Get available sections for a document from ChromaDB.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of available section names
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return []
            
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )
            
            sections = []
            if results and "metadatas" in results:
                for metadata in results["metadatas"]:
                    sections.append(metadata["section"])
            
            return list(set(sections))
            
        except Exception as e:
            logger.error(f"Error retrieving sections from ChromaDB for {doc_id}: {str(e)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its embeddings from ChromaDB.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return False
            
        try:
            # Get all embeddings for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )
            
            if results and "ids" in results and results["ids"]:
                # Delete all embeddings
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted document {doc_id} with {len(results['ids'])} embeddings from ChromaDB")
                return True
            else:
                logger.warning(f"No embeddings found in ChromaDB for document {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document from ChromaDB {doc_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if self.collection is None:
                return {"error": "Collection not initialized"}
            
            # Get collection count
            count = self.collection.count()
            
            # Get unique document types
            try:
                results = self.collection.get(
                    include=["metadatas"],
                    limit=count
                )
                
                if results and results["metadatas"]:
                    doc_types = {}
                    for metadata in results["metadatas"]:
                        if metadata and "doc_type" in metadata:
                            doc_type = metadata["doc_type"]
                            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                else:
                    doc_types = {}
                    
            except Exception as e:
                logger.warning(f"Could not get document type statistics: {str(e)}")
                doc_types = {}
            
            return {
                "total_documents": count,
                "document_types": doc_types,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": f"Error getting collection stats: {str(e)}"}
    
    def get_all_documents(self, doc_type: str = None) -> List[Dict]:
        """
        Get all documents from the collection.
        
        Args:
            doc_type: Filter by document type (cv, job, or None for all)
            
        Returns:
            List of all documents with their metadata
        """
        try:
            if self.collection is None:
                return []
            
            # Get all documents
            count = self.collection.count()
            if count == 0:
                return []
            
            # Get documents with metadata
            results = self.collection.get(
                include=["metadatas", "documents"],
                limit=count
            )
            
            documents = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    
                    # Filter by document type if specified
                    if doc_type and metadata.get("doc_type") != doc_type:
                        continue
                    
                    documents.append({
                        "doc_id": doc_id,
                        "doc_type": metadata.get("doc_type"),
                        "raw_text": results["documents"][i] if results["documents"] else "",
                        "metadata": metadata
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            return []
    
    def clear_collection(self) -> bool:
        """
        Clear all data from the ChromaDB collection.
        
        Returns:
            True if successful, False otherwise
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return False
            
        try:
            self.collection.delete(where={})
            logger.info("Cleared all data from ChromaDB collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing ChromaDB collection: {str(e)}")
            return False

# Global ChromaDB engine instance
chroma_engine = ChromaEngine() 