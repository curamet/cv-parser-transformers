# Models package for semantic CV-Job matching system 
# This package contains data models, schemas, and ML model configurations

# Data models for API requests/responses
from .data_models import *

# Domain models for business entities
from .domain_models import *

# ML model configurations
from .ml_models import *

__all__ = [
    # Data models
    'CVUploadRequest',
    'JobUploadRequest', 
    'MatchingRequest',
    'MatchingResponse',
    
    # Domain models
    'Document',
    'CV',
    'Job',
    'Match',
    
    # ML models
    'EmbeddingModel',
    'SkillCategorizerModel'
] 