# Database package for semantic CV-Job matching system 
# This package contains database interfaces and abstractions

from .vector_db import vector_db, VectorDatabase

__all__ = [
    'vector_db', 'VectorDatabase'
] 