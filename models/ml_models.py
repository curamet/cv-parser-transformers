"""
ML model configurations and settings.
These models define the structure of machine learning models used in the system.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class ModelType(str, Enum):
    """Types of ML models."""
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    ZERO_SHOT = "zero_shot"
    TRANSFORMER = "transformer"

class EmbeddingModel(BaseModel):
    """Embedding model configuration."""
    name: str = Field(..., description="Model name")
    type: ModelType = Field(default=ModelType.EMBEDDING, description="Model type")
    model_path: str = Field(..., description="Model path or identifier")
    dimensions: int = Field(..., description="Embedding dimensions")
    max_length: int = Field(..., description="Maximum input length")
    device: str = Field(default="cpu", description="Device to run model on")
    batch_size: int = Field(default=32, description="Batch size for processing")
    
    # Model-specific parameters
    parameters: Dict[str, Any] = Field(default={}, description="Model-specific parameters")
    
    class Config:
        use_enum_values = True

class SkillCategorizerModel(BaseModel):
    """Skill categorization model configuration."""
    name: str = Field(..., description="Model name")
    type: ModelType = Field(default=ModelType.ZERO_SHOT, description="Model type")
    model_path: str = Field(..., description="Model path or identifier")
    categories: List[str] = Field(..., description="Skill categories")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold")
    device: str = Field(default="cpu", description="Device to run model on")
    
    # Categorization parameters
    parameters: Dict[str, Any] = Field(default={}, description="Model-specific parameters")
    
    class Config:
        use_enum_values = True

class NLPProcessorModel(BaseModel):
    """NLP processor model configuration."""
    name: str = Field(..., description="Model name")
    type: ModelType = Field(default=ModelType.TRANSFORMER, description="Model type")
    model_path: str = Field(..., description="Model path or identifier")
    task: str = Field(..., description="NLP task (e.g., 'text-classification', 'token-classification')")
    device: str = Field(default="cpu", description="Device to run model on")
    
    # Processing parameters
    parameters: Dict[str, Any] = Field(default={}, description="Model-specific parameters")
    
    class Config:
        use_enum_values = True

class ModelRegistry(BaseModel):
    """Model registry containing all ML models."""
    embedding_model: EmbeddingModel = Field(..., description="Embedding model configuration")
    skill_categorizer: SkillCategorizerModel = Field(..., description="Skill categorizer configuration")
    nlp_processor: Optional[NLPProcessorModel] = Field(None, description="NLP processor configuration")
    
    # Additional models
    additional_models: Dict[str, Any] = Field(default={}, description="Additional model configurations")

class ModelPerformance(BaseModel):
    """Model performance metrics."""
    model_name: str = Field(..., description="Model name")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    precision: Optional[float] = Field(None, description="Model precision")
    recall: Optional[float] = Field(None, description="Model recall")
    f1_score: Optional[float] = Field(None, description="F1 score")
    inference_time: Optional[float] = Field(None, description="Average inference time in seconds")
    memory_usage: Optional[float] = Field(None, description="Memory usage in MB")
    last_updated: Optional[str] = Field(None, description="Last performance update timestamp")

class ModelConfig(BaseModel):
    """Global model configuration."""
    registry: ModelRegistry = Field(..., description="Model registry")
    performance: Dict[str, ModelPerformance] = Field(default={}, description="Performance metrics by model")
    auto_update: bool = Field(default=False, description="Auto-update models")
    cache_models: bool = Field(default=True, description="Cache models in memory")
    fallback_models: Dict[str, str] = Field(default={}, description="Fallback model mappings") 