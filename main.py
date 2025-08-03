#!/usr/bin/env python3
"""
Semantic CV-Job Matching System
A comprehensive system for intelligent CV and job opportunity matching using HuggingFace transformers.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger
import tempfile
import shutil
from utils.llm import analyze_cv_with_llm, parse_cv_analysis
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from services.document_processor import document_processor
from services.semantic_processor import semantic_processor
from database.vector_db import vector_db

# Configure logging
logger.add(
    settings.LOGGING_SETTINGS["file"],
    level=settings.LOGGING_SETTINGS["level"],
    format=settings.LOGGING_SETTINGS["format"],
    rotation="10 MB",
    retention="7 days"
)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    try:
        logger.info("🚀 Starting CV-Job Matching System...")
        
        # Pre-load models to avoid delays on first request
        logger.info("📚 Pre-loading models...")
        
        # Preload the main NLP model
        from utils.model_cache import model_cache
        model_cache.preload_models()
        
        # Pre-load skill categorizer model
        from utils.skill_categorizer import skill_categorizer
        # Trigger model loading by calling categorize_skills with empty list
        skill_categorizer.categorize_skills([])
        logger.info("✅ All models pre-loaded successfully")
        
        logger.info("✅ System startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down CV-Job Matching System...")

# Create FastAPI app
app = FastAPI(
    title="Semantic CV-Job Matching System",
    description="Intelligent CV and job opportunity matching using HuggingFace transformers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class ProcessCVsRequest(BaseModel):
    cv_files: List[str]

class ProcessJobsRequest(BaseModel):
    job_files: List[str]

class ProcessMultipleCVsRequest(BaseModel):
    cv_files_with_ids: List[Dict[str, str]]  # [{"file_path": "path/to/cv.pdf", "cv_id": "cv_123"}]

class ProcessMultipleJobsRequest(BaseModel):
    job_files_with_ids: List[Dict[str, str]]  # [{"file_path": "path/to/job.pdf", "job_id": "job_456"}]

class ProcessMultipleOpportunitiesRequest(BaseModel):
    opportunities: List[Dict[str, str]]  # [{"opportunity_id": "job_789", "opportunity_text": "...", "opportunity_title": "..."}]

class ProcessSingleCVRequest(BaseModel):
    cv_id: str
    cv_file: UploadFile

class ProcessOpportunityRequest(BaseModel):
    opportunity_id: str
    opportunity_text: str
    opportunity_title: Optional[str] = None

class MatchSingleCVWithOpportunityRequest(BaseModel):
    cv_id: str
    opportunity_id: str

class MatchOpportunityRequest(BaseModel):
    opportunity_id: str
    cv_ids: List[str]

class MatchCVRequest(BaseModel):
    cv_id: str
    opportunity_ids: List[str]

class MatchMultipleCVsWithOpportunitiesRequest(BaseModel):
    cv_ids: List[str]
    opportunity_ids: List[str]

class MatchCVWithAllOpportunitiesRequest(BaseModel):
    cv_id: str
    n_results: int = 10

class MatchOpportunityWithAllCVsRequest(BaseModel):
    opportunity_id: str
    n_results: int = 10

class SearchRequest(BaseModel):
    query: str
    doc_type: Optional[str] = None
    n_results: int = 10

class HealthResponse(BaseModel):
    status: str
    message: str
    details: Dict

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Semantic CV-Job Matching System",
        "version": "1.0.0",
        "description": "Intelligent CV and job opportunity matching using HuggingFace transformers",
        # "endpoints": {
        #     "process_cvs": "/process-cvs",
        #     "process_jobs": "/process-jobs",
        #     "match_opportunity": "/match-opportunity",
        #     "match_cv": "/match-cv",
        #     "search": "/search",
        #     "health": "/health",
        #     "stats": "/stats"
        # }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector database health
        db_health = vector_db.health_check()
        
        # Check NLP processor
        nlp_health = "healthy"
        try:
            from services.nlp_processor import nlp_processor
            # Test embedding generation
            test_embedding = nlp_processor.generate_embeddings("test")
            if test_embedding is None or len(test_embedding) == 0:
                nlp_health = "unhealthy"
        except Exception as e:
            nlp_health = "unhealthy"
            logger.error(f"NLP processor health check failed: {str(e)}")
        
        # Overall system health
        overall_status = "healthy"
        if db_health["status"] == "unhealthy" or nlp_health == "unhealthy":
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            message="System health check completed",
            details={
                "vector_database": db_health,
                "nlp_processor": nlp_health,
                "model": settings.get_model_name()
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            message="Health check failed",
            details={"error": str(e)}
        )

@app.post("/process-cvs", tags=["Processing"])
async def process_cvs(background_tasks: BackgroundTasks, request: ProcessCVsRequest):
    """
    Process multiple CV files and store them in the vector database.
    
    Args:
        request: List of CV file paths to process
        
    Returns:
        Processing results with CV IDs and analysis
    """
    try:
        logger.info(f"Processing {len(request.cv_files)} CV files")
        
        # Validate files exist
        valid_files = []
        for file_path in request.cv_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"File not found: {file_path}")
        
        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid CV files found")
        
        # Process CVs
        processed_cvs = document_processor.process_documents_pipeline(valid_files, "cv")
        
        # Extract CV IDs and summaries
        cv_summaries = []
        for cv_data in processed_cvs:
            cv_id = cv_data["doc_id"]
            summary = {
                "cv_id": cv_id,
                "file_name": cv_data["file_name"],
                "analysis": cv_data.get("analysis", {}),
                "sections_found": list(cv_data.get("sections", {}).keys())
            }
            cv_summaries.append(summary)
        
        return {
            "message": f"Successfully processed {len(processed_cvs)} CVs",
            "processed_count": len(processed_cvs),
            "cv_summaries": cv_summaries
        }
        
    except Exception as e:
        logger.error(f"Error processing CVs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing CVs: {str(e)}")

@app.post("/process-jobs", tags=["Processing"])
async def process_jobs(background_tasks: BackgroundTasks, request: ProcessJobsRequest):
    """
    Process multiple job description files and store them in the vector database.
    
    Args:
        request: List of job file paths to process
        
    Returns:
        Processing results with job IDs and analysis
    """
    try:
        logger.info(f"Processing {len(request.job_files)} job files")
        
        # Validate files exist
        valid_files = []
        for file_path in request.job_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"File not found: {file_path}")
        
        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid job files found")
        
        # Process jobs
        processed_jobs = document_processor.process_documents_pipeline(valid_files, "job")
        
        # Extract job IDs and summaries
        job_summaries = []
        for job_data in processed_jobs:
            job_id = job_data["doc_id"]
            summary = {
                "job_id": job_id,
                "file_name": job_data["file_name"],
                "analysis": job_data.get("analysis", {}),
                "sections_found": list(job_data.get("sections", {}).keys())
            }
            job_summaries.append(summary)
        
        return {
            "message": f"Successfully processed {len(processed_jobs)} jobs",
            "processed_count": len(processed_jobs),
            "job_summaries": job_summaries
        }
        
    except Exception as e:
        logger.error(f"Error processing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing jobs: {str(e)}")

@app.post("/process-multiple-cvs", tags=["Processing"])
async def process_multiple_cvs_with_custom_ids(background_tasks: BackgroundTasks, request: ProcessMultipleCVsRequest):
    """
    Process multiple CV files with custom IDs and store them in the vector database.
    
    Args:
        request: List of CV files with their custom IDs
        
    Returns:
        Processing results with CV IDs and analysis
    """
    try:
        logger.info(f"Processing {len(request.cv_files_with_ids)} CV files with custom IDs")
        
        # Validate files exist and extract file paths
        valid_files = []
        file_id_mapping = {}
        
        for cv_data in request.cv_files_with_ids:
            file_path = cv_data.get("file_path")
            cv_id = cv_data.get("cv_id")
            
            if not file_path or not cv_id:
                logger.warning(f"Missing file_path or cv_id in request: {cv_data}")
                continue
                
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            valid_files.append(file_path)
            file_id_mapping[file_path] = cv_id
        
        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid CV files found")
        
        # Process CVs with custom IDs
        custom_ids = [file_id_mapping[file_path] for file_path in valid_files]
        logger.info(f"Processing {len(valid_files)} CVs with custom IDs: {custom_ids}")
        
        # Process all files with custom IDs
        processed_cvs = document_processor.process_documents_pipeline(valid_files, "cv", custom_ids=custom_ids)
        
        # Extract CV IDs and summaries
        cv_summaries = []
        for cv_data in processed_cvs:
            cv_id = cv_data["doc_id"]
            summary = {
                "cv_id": cv_id,
                "file_name": cv_data["file_name"],
                "analysis": cv_data.get("analysis", {}),
                "sections_found": list(cv_data.get("sections", {}).keys())
            }
            cv_summaries.append(summary)
        
        return {
            "message": f"Successfully processed {len(processed_cvs)} CVs with custom IDs",
            "processed_count": len(processed_cvs),
            "cv_summaries": cv_summaries
        }
        
    except Exception as e:
        logger.error(f"Error processing multiple CVs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing multiple CVs: {str(e)}")

@app.post("/process-multiple-jobs", tags=["Processing"])
async def process_multiple_jobs_with_custom_ids(background_tasks: BackgroundTasks, request: ProcessMultipleJobsRequest):
    """
    Process multiple job files with custom IDs and store them in the vector database.
    
    Args:
        request: List of job files with their custom IDs
        
    Returns:
        Processing results with job IDs and analysis
    """
    try:
        logger.info(f"Processing {len(request.job_files_with_ids)} job files with custom IDs")
        
        # Validate files exist and extract file paths
        valid_files = []
        file_id_mapping = {}
        
        for job_data in request.job_files_with_ids:
            file_path = job_data.get("file_path")
            job_id = job_data.get("job_id")
            
            if not file_path or not job_id:
                logger.warning(f"Missing file_path or job_id in request: {job_data}")
                continue
                
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            valid_files.append(file_path)
            file_id_mapping[file_path] = job_id
        
        if not valid_files:
            raise HTTPException(status_code=400, detail="No valid job files found")
        
        # Process jobs with custom IDs
        custom_ids = [file_id_mapping[file_path] for file_path in valid_files]
        logger.info(f"Processing {len(valid_files)} jobs with custom IDs: {custom_ids}")
        
        # Process all files with custom IDs
        processed_jobs = document_processor.process_documents_pipeline(valid_files, "job", custom_ids=custom_ids)
        
        # Extract job IDs and summaries
        job_summaries = []
        for job_data in processed_jobs:
            job_id = job_data["doc_id"]
            summary = {
                "job_id": job_id,
                "file_name": job_data["file_name"],
                "analysis": job_data.get("analysis", {}),
                "sections_found": list(job_data.get("sections", {}).keys())
            }
            job_summaries.append(summary)
        
        return {
            "message": f"Successfully processed {len(processed_jobs)} jobs with custom IDs",
            "processed_count": len(processed_jobs),
            "job_summaries": job_summaries
        }
        
    except Exception as e:
        logger.error(f"Error processing multiple jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing multiple jobs: {str(e)}")

@app.post("/process-multiple-opportunities", tags=["Processing"])
async def process_multiple_opportunities_with_custom_ids(background_tasks: BackgroundTasks, request: ProcessMultipleOpportunitiesRequest):
    """
    Process multiple opportunities with custom IDs and store them in the vector database.
    
    Args:
        request: List of opportunities with their custom IDs
        
    Returns:
        Processing results with opportunity IDs and analysis
    """
    try:
        logger.info(f"Processing {len(request.opportunities)} opportunities with custom IDs")
        
        processed_opportunities = []
        
        for opportunity_data in request.opportunities:
            opportunity_id = opportunity_data.get("opportunity_id")
            opportunity_text = opportunity_data.get("opportunity_text")
            opportunity_title = opportunity_data.get("opportunity_title", "Opportunity")
            
            if not opportunity_id or not opportunity_text:
                logger.warning(f"Missing opportunity_id or opportunity_text in request: {opportunity_data}")
                continue
            
            logger.info(f"Processing opportunity with custom ID: {opportunity_id}")
            
            # Process opportunity text with custom ID
            processed_job = document_processor.process_text_directly(
                opportunity_text, 
                "job",
                opportunity_title,
                custom_id=opportunity_id
            )
            
            if processed_job:
                processed_opportunities.append(processed_job)
        
        # Extract opportunity IDs and summaries
        opportunity_summaries = []
        for job_data in processed_opportunities:
            job_id = job_data["doc_id"]
            summary = {
                "job_id": job_id,
                "title": job_data.get("file_name", "Opportunity"),
                "analysis": job_data.get("analysis", {}),
                "sections_found": list(job_data.get("sections", {}).keys())
            }
            opportunity_summaries.append(summary)
        
        return {
            "message": f"Successfully processed {len(processed_opportunities)} opportunities with custom IDs",
            "processed_count": len(processed_opportunities),
            "opportunity_summaries": opportunity_summaries
        }
        
    except Exception as e:
        logger.error(f"Error processing multiple opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing multiple opportunities: {str(e)}")

@app.post("/process-single-cv", tags=["Processing"])
async def process_single_cv(cv_id: str = Form(...), cv_file: UploadFile = File(...)):
    """
    Process a single CV file uploaded via form and store it in the vector database with custom ID.
    
    Args:
        cv_id: Custom ID for the CV
        cv_file: CV file uploaded via form
        
    Returns:
        Processing results with CV ID and analysis
    """
    try:
        logger.info(f"Processing single CV file: {cv_file.filename} with ID: {cv_id}")
        
        # Validate file
        if not cv_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        file_size = 0
        content = await cv_file.read()
        file_size = len(content)
        
        max_size = settings.FILE_SETTINGS["max_file_size_mb"] * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {settings.FILE_SETTINGS['max_file_size_mb']}MB")
        
        # Check file format
        file_extension = Path(cv_file.filename).suffix.lower()
        if file_extension not in settings.FILE_SETTINGS["supported_formats"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
        
        # Reset file position for processing
        cv_file.file.seek(0)
        
        logger.info(f"Starting CV processing pipeline...")
        
        # Process the CV directly using file object with custom ID
        processed_cvs = document_processor.process_documents_pipeline([cv_file], "cv", custom_id=cv_id)
        
        logger.info(f"Processing pipeline completed. Got {len(processed_cvs) if processed_cvs else 0} processed CVs")
        
        if not processed_cvs:
            raise HTTPException(status_code=500, detail="Failed to process CV file")
        
        processed_cv = processed_cvs[0]
        stored_cv_id = processed_cv["doc_id"]
        
        logger.info(f"Successfully processed CV with ID: {stored_cv_id}")
        
        # Create summary
        summary = {
            "cv_id": stored_cv_id,
            "file_name": cv_file.filename,
            "analysis": processed_cv.get("analysis", {}),
            "sections_found": list(processed_cv.get("sections", {}).keys())
        }
        
        logger.info(f"Preparing response with summary: {summary}")
        
        response = {
            "message": "Successfully processed CV",
            "cv_summary": summary
        }
        
        logger.info(f"Returning response: {response}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing single CV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")

@app.post("/process-opportunity", tags=["Processing"])
async def process_opportunity(request: ProcessOpportunityRequest):
    """
    Process opportunity text directly from request and store it in the vector database with custom ID.
    
    Args:
        request: Opportunity ID, text and optional title
        
    Returns:
        Processing results with opportunity ID and analysis
    """
    try:
        logger.info(f"Processing opportunity text from request with ID: {request.opportunity_id}")
        
        if not request.opportunity_text.strip():
            raise HTTPException(status_code=400, detail="Opportunity text cannot be empty")
        
        if not request.opportunity_id.strip():
            raise HTTPException(status_code=400, detail="Opportunity ID cannot be empty")
        
        # Process the opportunity text directly with custom ID
        processed_job = document_processor.process_text_directly(
            request.opportunity_text, 
            "job",
            request.opportunity_title or "Opportunity",
            custom_id=request.opportunity_id
        )
        
        if not processed_job:
            raise HTTPException(status_code=500, detail="Failed to process opportunity text")
        
        job_id = processed_job["doc_id"]
        
        # Create summary
        summary = {
            "job_id": job_id,
            "title": request.opportunity_title or "Opportunity",
            "analysis": processed_job.get("analysis", {}),
            "sections_found": list(processed_job.get("sections", {}).keys())
        }
        
        return {
            "message": "Successfully processed opportunity",
            "job_summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing opportunity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing opportunity: {str(e)}")

@app.post("/match-opportunity", tags=["Matching"])
async def match_opportunity(request: MatchOpportunityRequest):
    """
    Match a job opportunity with a list of CVs.
    
    Args:
        request: Opportunity ID and list of CV IDs to match against
        
    Returns:
        Matching results with scores and explanations
    """
    try:
        logger.info(f"Matching opportunity {request.opportunity_id} with {len(request.cv_ids)} CVs")
        
        # Perform matching
        results = semantic_processor.match_opportunity_with_cvs(
            request.opportunity_id, 
            request.cv_ids
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return results
        
    except Exception as e:
        logger.error(f"Error matching opportunity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error matching opportunity: {str(e)}")

@app.post("/match-cv", tags=["Matching"])
async def match_cv(request: MatchCVRequest):
    """
    Find matching opportunities for a given CV.
    
    Args:
        request: CV ID and list of opportunity IDs to match against
        
    Returns:
        Matching results with scores and explanations
    """
    try:
        logger.info(f"Finding opportunities for CV {request.cv_id} from {len(request.opportunity_ids)} opportunities")
        
        # Perform matching
        results = semantic_processor.match_cv_with_opportunities(
            request.cv_id, 
            request.opportunity_ids
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return results
        
    except Exception as e:
        logger.error(f"Error matching CV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error matching CV: {str(e)}")

@app.post("/match-multiple-cvs-with-opportunities", tags=["Matching"])
async def match_multiple_cvs_with_opportunities(request: MatchMultipleCVsWithOpportunitiesRequest):
    """
    Match multiple CVs with multiple opportunities.
    
    Args:
        request: List of CV IDs and opportunity IDs to match
        
    Returns:
        Matching results with scores and explanations
    """
    try:
        logger.info(f"Matching {len(request.cv_ids)} CVs with {len(request.opportunity_ids)} opportunities")
        
        # Perform matching
        results = semantic_processor.match_multiple_cvs_with_opportunities(
            request.cv_ids, 
            request.opportunity_ids
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return results
        
    except Exception as e:
        logger.error(f"Error matching multiple CVs with opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error matching multiple CVs with opportunities: {str(e)}")

@app.post("/match-cv-with-all-opportunities", tags=["Matching"])
async def match_cv_with_all_opportunities(request: MatchCVWithAllOpportunitiesRequest):
    """
    Find matching opportunities for a given CV, returning all matches.
    
    Args:
        request: CV ID and number of results to return
        
    Returns:
        Matching results with scores and explanations for all opportunities
    """
    try:
        logger.info(f"Finding all opportunities for CV {request.cv_id} with {request.n_results} results")
        
        # Perform matching
        results = semantic_processor.match_cv_with_opportunities(
            request.cv_id, 
            [], # Match against all opportunities
            n_results=request.n_results
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return results
        
    except Exception as e:
        logger.error(f"Error matching CV with all opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error matching CV with all opportunities: {str(e)}")

@app.post("/match-opportunity-with-all-cvs", tags=["Matching"])
async def match_opportunity_with_all_cvs(request: MatchOpportunityWithAllCVsRequest):
    """
    Find matching CVs for a given opportunity, returning all matches.
    
    Args:
        request: Opportunity ID and number of results to return
        
    Returns:
        Matching results with scores and explanations for all CVs
    """
    try:
        logger.info(f"Finding all CVs for opportunity {request.opportunity_id} with {request.n_results} results")
        
        # Perform matching
        results = semantic_processor.match_opportunity_with_cvs(
            request.opportunity_id, 
            [], # Match against all CVs
            n_results=request.n_results
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return results
        
    except Exception as e:
        logger.error(f"Error matching opportunity with all CVs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error matching opportunity with all CVs: {str(e)}")

@app.post("/match-single-cv-with-opportunity", tags=["Matching"])
async def match_single_cv_with_opportunity(request: MatchSingleCVWithOpportunityRequest):
    """
    Match a single CV with an opportunity using their stored IDs.
    
    Args:
        request: CV ID and opportunity ID
        
    Returns:
        Matching results with scores and explanations
    """
    try:
        logger.info(f"Matching CV (ID: {request.cv_id}) with opportunity (ID: {request.opportunity_id})")
        
        # Validate inputs
        if not request.cv_id.strip():
            raise HTTPException(status_code=400, detail="CV ID cannot be empty")
        
        if not request.opportunity_id.strip():
            raise HTTPException(status_code=400, detail="Opportunity ID cannot be empty")
        
        # Check if documents exist in database
        cv_exists = vector_db.get_document(request.cv_id)
        if cv_exists is None:
            raise HTTPException(status_code=404, detail=f"CV with ID {request.cv_id} not found in database")
        
        opportunity_exists = vector_db.get_document(request.opportunity_id)
        if opportunity_exists is None:
            raise HTTPException(status_code=404, detail=f"Opportunity with ID {request.opportunity_id} not found in database")
        
        # Perform matching using stored documents
        results = semantic_processor.match_opportunity_with_cvs(request.opportunity_id, [request.cv_id])
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        # Add ID information to results
        results["cv_id"] = request.cv_id
        results["opportunity_id"] = request.opportunity_id
        
        return results
        
    except Exception as e:
        logger.error(f"Error matching single CV with opportunity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error matching single CV with opportunity: {str(e)}")

@app.get("/llm/analyse-cv/{doc_id}", tags=["Analysis"])
async def analyse_cv(doc_id: str):
    """
    Analyse a CV document using LLM

    Args:
        doc_id: Document ID to analyse

    Returns:
        Analysis results
    """
    try:
        # Get CV text
        cv_text = vector_db.get_raw_text(doc_id)

        if not cv_text:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # Get the full prompt and analyze with LLM
        analysis_response = await analyze_cv_with_llm(cv_text)

        if not analysis_response:
            raise HTTPException(status_code=500, detail="Failed to get analysis from LLM")


        parsed_analysis = parse_cv_analysis(analysis_response)

        logger.info(f"Successfully analyzed CV {doc_id}")

        return {
            "message": f"Successfully analysed document {doc_id}",
            "doc_id": doc_id,
            "analysis": parsed_analysis,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analysing document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analysing document: {str(e)}")

@app.post("/search", tags=["Search"])
async def search_documents(request: SearchRequest):
    """
    Search for similar documents using semantic similarity.
    
    Args:
        request: Search query and parameters
        
    Returns:
        List of similar documents with scores
    """
    try:
        logger.info(f"Searching for documents similar to: {request.query}")
        
        # Perform search
        results = vector_db.search_similar(
            query_text=request.query,
            doc_type=request.doc_type or "all",
            n_results=request.n_results
        )
        
        return {
            "query": request.query,
            "doc_type": request.doc_type,
            "n_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.get("/search/cvs/{query}", tags=["Search"])
async def search_cvs_by_query(query: str, n_results: int = 10):
    """
    Search for CVs using semantic search and return only CV IDs.
    
    Args:
        query: Search query (e.g., "python developer")
        n_results: Number of results to return
        
    Returns:
        List of CV IDs that match the query
    """
    try:
        logger.info(f"Searching CVs with query: {query}")
        
        # Perform semantic search for CVs
        cv_ids = semantic_processor.search_cvs_by_query(query, n_results)
        
        return {
            "query": query,
            "doc_type": "cv",
            "n_results": len(cv_ids),
            "cv_ids": cv_ids
        }
        
    except Exception as e:
        logger.error(f"Error searching CVs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching CVs: {str(e)}")

@app.get("/search/opportunities/{query}", tags=["Search"])
async def search_opportunities_by_query(query: str, n_results: int = 10):
    """
    Search for opportunities/jobs using semantic search and return only opportunity IDs.
    
    Args:
        query: Search query (e.g., "python developer")
        n_results: Number of results to return
        
    Returns:
        List of opportunity IDs that match the query
    """
    try:
        logger.info(f"Searching opportunities with query: {query}")
        
        # Perform semantic search for opportunities
        opportunity_ids = semantic_processor.search_opportunities_by_query(query, n_results)
        
        return {
            "query": query,
            "doc_type": "job",
            "n_results": len(opportunity_ids),
            "opportunity_ids": opportunity_ids
        }
        
    except Exception as e:
        logger.error(f"Error searching opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching opportunities: {str(e)}")

@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """
    Get system statistics and database information.
    
    Returns:
        System statistics
    """
    try:
        stats = vector_db.get_statistics()
        
        return {
            "system_stats": {
                "model": settings.get_model_name(),
                "embedding_dimension": settings.get_embedding_dimension(settings.get_model_name()),
                "scoring_weights": settings.SCORING_WEIGHTS,
                "matching_thresholds": settings.MATCHING_THRESHOLDS
            },
            "database_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@app.get("/documents/{doc_id}", tags=["Management"])
async def get_document_content(doc_id: str):
    """
    Get the raw text content of a document from the vector database.
    
    Args:
        doc_id: Document ID to retrieve
        
    Returns:
        Document content and metadata
    """
    try:
        # Get raw text content
        raw_text = vector_db.get_raw_text(doc_id)
        
        if raw_text is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Get document sections
        sections = vector_db.get_document_sections(doc_id)
        
        # Get document statistics
        stats = vector_db.get_statistics()
        
        return {
            "doc_id": doc_id,
            "raw_text": raw_text,
            "sections": sections,
            "text_length": len(raw_text),
            "database_stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@app.delete("/documents/{doc_id}", tags=["Management"])
async def delete_document(doc_id: str):
    """
    Delete a document from the vector database.
    
    Args:
        doc_id: Document ID to delete
        
    Returns:
        Deletion result
    """
    try:
        success = vector_db.delete_document(doc_id)
        
        if success:
            return {"message": f"Successfully deleted document {doc_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
            
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.delete("/clear-database", tags=["Management"])
async def clear_database():
    """
    Clear all data from the vector database.
    
    Returns:
        Clear operation result
    """
    try:
        success = vector_db.clear_database()
        
        if success:
            return {"message": "Successfully cleared all data from database"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear database")
            
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(settings.VECTOR_DB_SETTINGS["persist_directory"], exist_ok=True)
    os.makedirs(settings.FILE_SETTINGS["upload_directory"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=settings.API_SETTINGS["host"],
        port=settings.API_SETTINGS["port"],
        reload=settings.API_SETTINGS["debug"],
        workers=settings.API_SETTINGS["workers"]
    ) 