"""
Domain models for business entities.
These models represent the core business concepts and entities.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    """Document types."""
    CV = "cv"
    JOB = "job"

class MatchQuality(str, Enum):
    """Match quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNMATCHED = "unmatched"

class SeniorityLevel(str, Enum):
    """Seniority levels."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"

class Document(BaseModel):
    """Base document model."""
    doc_id: str = Field(..., description="Unique document identifier")
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")
    file_extension: str = Field(..., description="File extension")
    doc_type: DocumentType = Field(..., description="Document type")
    raw_text: str = Field(..., description="Extracted text content")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    
    class Config:
        use_enum_values = True

class CV(Document):
    """CV document model."""
    doc_type: DocumentType = Field(default=DocumentType.CV, description="Document type")
    
    # CV-specific fields
    contact_info: Optional[Dict[str, str]] = Field(None, description="Contact information")
    skills: Optional[List[str]] = Field(None, description="Extracted skills")
    experience_years: Optional[int] = Field(None, description="Years of experience")
    education_level: Optional[str] = Field(None, description="Highest education level")
    projects: Optional[List[str]] = Field(None, description="Project descriptions")

class Job(Document):
    """Job description model."""
    doc_type: DocumentType = Field(default=DocumentType.JOB, description="Document type")
    
    # Job-specific fields
    required_skills: Optional[List[str]] = Field(None, description="Required skills")
    preferred_skills: Optional[List[str]] = Field(None, description="Preferred skills")
    experience_requirement: Optional[int] = Field(None, description="Required years of experience")
    education_requirement: Optional[str] = Field(None, description="Required education level")
    responsibilities: Optional[List[str]] = Field(None, description="Job responsibilities")
    company: Optional[str] = Field(None, description="Company name")
    location: Optional[str] = Field(None, description="Job location")

class Skill(BaseModel):
    """Skill model."""
    name: str = Field(..., description="Skill name")
    category: str = Field(..., description="Skill category")
    confidence: float = Field(..., description="Confidence score")
    source: str = Field(..., description="Source of skill extraction")

class Experience(BaseModel):
    """Experience model."""
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    duration: str = Field(..., description="Duration of employment")
    description: str = Field(..., description="Job description")
    skills_used: List[str] = Field(default=[], description="Skills used in this role")

class Education(BaseModel):
    """Education model."""
    degree: str = Field(..., description="Degree obtained")
    institution: str = Field(..., description="Educational institution")
    field_of_study: str = Field(..., description="Field of study")
    graduation_year: Optional[int] = Field(None, description="Graduation year")
    gpa: Optional[float] = Field(None, description="Grade point average")

class Match(BaseModel):
    """Match model representing a CV-Job match."""
    cv_id: str = Field(..., description="CV identifier")
    job_id: str = Field(..., description="Job identifier")
    overall_score: float = Field(..., description="Overall match score")
    match_quality: MatchQuality = Field(..., description="Match quality level")
    section_scores: Dict[str, float] = Field(..., description="Scores by section")
    section_explanations: Dict[str, str] = Field(..., description="Explanations by section")
    overall_explanation: str = Field(..., description="Overall match explanation")
    matched_sections: List[str] = Field(..., description="Sections that were matched")
    timestamp: datetime = Field(default_factory=datetime.now, description="Match timestamp")
    
    class Config:
        use_enum_values = True

class AnalysisResult(BaseModel):
    """Document analysis result model."""
    doc_id: str = Field(..., description="Document identifier")
    skills_analysis: Dict[str, Any] = Field(..., description="Skills analysis results")
    experience_analysis: Dict[str, Any] = Field(..., description="Experience analysis results")
    education_analysis: Dict[str, Any] = Field(..., description="Education analysis results")
    overall_assessment: Dict[str, Any] = Field(..., description="Overall document assessment")
    semantic_features: Dict[str, Any] = Field(..., description="Semantic features extracted")
    embeddings: Dict[str, Any] = Field(..., description="Document embeddings")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp") 