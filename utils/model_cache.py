"""
Model caching utility to ensure models are loaded only once and shared efficiently.
"""

import os
import threading
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from config.settings import settings


class ModelCache:
    """Singleton model cache to prevent multiple model loads."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelCache, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._models: Dict[str, Any] = {}
            self._model_locks: Dict[str, threading.Lock] = {}
            self._initialized = True
    
    def get_sentence_transformer(self, model_name: str) -> SentenceTransformer:
        """
        Get or create a SentenceTransformer model instance.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            SentenceTransformer instance
        """
        if model_name not in self._model_locks:
            self._model_locks[model_name] = threading.Lock()
        
        with self._model_locks[model_name]:
            if model_name not in self._models:
                logger.info(f"Loading SentenceTransformer model: {model_name}")
                self._models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded SentenceTransformer model: {model_name}")
            
            return self._models[model_name]
    
    def get_huggingface_model(self, model_name: str) -> tuple[AutoTokenizer, AutoModel]:
        """
        Get or create HuggingFace model and tokenizer instances.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (tokenizer, model)
        """
        model_key = f"{model_name}_hf"
        
        if model_key not in self._model_locks:
            self._model_locks[model_key] = threading.Lock()
        
        with self._model_locks[model_key]:
            if model_key not in self._models:
                logger.info(f"Loading HuggingFace model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                self._models[model_key] = (tokenizer, model)
                logger.info(f"Successfully loaded HuggingFace model: {model_name}")
            
            return self._models[model_key]
    
    def preload_models(self):
        """Preload all required models at startup."""
        try:
            logger.info("🔄 Preloading models...")
            
            # Preload the primary sentence transformer model
            primary_model = settings.get_model_name()
            self.get_sentence_transformer(primary_model)
            
            logger.info("✅ Models preloaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error preloading models: {str(e)}")
            raise


# Global model cache instance
model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    return model_cache 