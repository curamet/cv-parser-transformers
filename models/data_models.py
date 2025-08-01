"""
Data models for API requests and responses.
These models define the structure of data exchanged with the API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class CVUploadRequest(BaseModel):
    """Request model for CV upload."""
    file: Any = Field(..., description="CV file to upload")
    title: Optional[str] = Field(None, description="Optional title for the CV")

class JobUploadRequest(BaseModel):
    """Request model for job upload."""
    file: Any = Field(..., description="Job description file to upload")
    title: Optional[str] = Field(None, description="Optional title for the job")

class TextUploadRequest(BaseModel):
    """Request model for text-based upload."""
    text: str = Field(..., description="Document text content")
    title: str = Field(..., description="Document title")
    doc_type: str = Field(..., description="Document type: 'cv' or 'job'")

class MatchingRequest(BaseModel):
    """Request model for matching operations."""
    cv_ids: Optional[List[str]] = Field(None, description="List of CV IDs to match")
    job_ids: Optional[List[str]] = Field(None, description="List of job IDs to match")
    match_type: str = Field(..., description="Type of matching: 'cv_to_jobs' or 'job_to_cvs'")

class MatchingResponse(BaseModel):
    """Response model for matching results."""
    request_id: str = Field(..., description="Unique request identifier")
    match_type: str = Field(..., description="Type of matching performed")
    matches: List[Dict[str, Any]] = Field(..., description="List of matches with scores")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class ProcessingResponse(BaseModel):
    """Response model for document processing."""
    doc_id: str = Field(..., description="Processed document ID")
    doc_type: str = Field(..., description="Document type")
    file_name: str = Field(..., description="Original file name")
    processing_status: str = Field(..., description="Processing status")
    analysis: Dict[str, Any] = Field(..., description="Document analysis results")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")

class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component statuses") 