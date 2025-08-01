from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
from services.document_parser import document_parser
from services.nlp_processor import nlp_processor
from database.vector_db import vector_db
from utils.text_processing import text_processor

class DocumentProcessor:
    """Unified document processing pipeline for CVs and job descriptions."""
    
    def __init__(self):
        """Initialize document processor."""
        pass
    
    def process_documents_pipeline(self, files: List, doc_type: str, custom_id: str = None, custom_ids: List[str] = None) -> List[Dict]:
        """
        Process multiple documents through the complete pipeline.
        
        Args:
            files: List of file objects (UploadFile, file-like objects, or file paths)
            doc_type: Type of document ("cv" or "job")
            custom_id: Optional custom ID for single file processing
            custom_ids: Optional list of custom IDs for multiple file processing
            
        Returns:
            List of processed document data with embeddings and analysis
        """
        logger.info(f"Starting {doc_type} processing pipeline for {len(files)} files")
        
        # Step 1: Parse documents
        if doc_type == "cv":
            parsed_docs = document_parser.process_multiple_cvs(files)
        else:
            parsed_docs = document_parser.process_multiple_jobs(files)
            
        if not parsed_docs:
            logger.error(f"No {doc_type}s were successfully parsed")
            return []
        
        # Step 2: Process each document
        processed_docs = []
        for i, doc_data in enumerate(parsed_docs):
            try:
                # Use custom ID if provided and this is the first (and only) document
                if custom_id and i == 0:
                    doc_data["doc_id"] = custom_id
                    logger.info(f"Using custom ID: {custom_id}")
                # Use custom IDs list if provided
                elif custom_ids and i < len(custom_ids):
                    doc_data["doc_id"] = custom_ids[i]
                    logger.info(f"Using custom ID: {custom_ids[i]}")
                
                processed_doc = self._process_single_document(doc_data, doc_type)
                if processed_doc:
                    processed_docs.append(processed_doc)
            except Exception as e:
                logger.error(f"Error processing {doc_type} {doc_data.get('doc_id', 'unknown')}: {str(e)}")
                continue
        
        # Step 3: Store vectors
        if processed_docs:
            stored_ids = vector_db.store_documents(processed_docs)
            logger.info(f"Stored {len(stored_ids)} {doc_type} vectors in database")
        
        logger.info(f"{doc_type.capitalize()} processing pipeline completed. Processed {len(processed_docs)} documents")
        return processed_docs
    
    def process_text_directly(self, text: str, doc_type: str, title: str = "Document", custom_id: str = None) -> Optional[Dict]:
        """
        Process document text directly without creating temporary files.
        
        Args:
            text: The document text content
            doc_type: Type of document ("cv" or "job")
            title: Optional title for the document
            custom_id: Optional custom ID for the document
            
        Returns:
            Processed document data with embeddings and analysis
        """
        logger.info(f"Processing {doc_type} text: {title}")
        
        try:
            # Use custom ID if provided, otherwise generate from text content
            if custom_id:
                doc_id = custom_id
                logger.info(f"Using custom ID: {custom_id}")
            else:
                import hashlib
                doc_id = hashlib.md5(text.encode()).hexdigest()
            
            # Create document data structure
            doc_data = {
                "doc_id": doc_id,
                "file_path": None,  # No file path since it's text
                "file_name": f"{title}.txt",
                "file_size": len(text.encode()),
                "file_extension": ".txt",
                "doc_type": doc_type,
                "raw_text": text,
                "processed_at": None
            }
            
            # Process the document
            processed_doc = self._process_single_document(doc_data, doc_type)
            if not processed_doc:
                logger.error(f"Failed to process {doc_type} text")
                return None
            
            # Store vector
            stored_ids = vector_db.store_documents([processed_doc])
                
            if not stored_ids:
                logger.error(f"Failed to store {doc_type} vector")
                return None
            
            logger.info(f"Successfully processed {doc_type} text: {doc_id}")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing {doc_type} text: {str(e)}")
            return None
    
    def _process_single_document(self, doc_data: Dict, doc_type: str) -> Optional[Dict]:
        """
        Process a single document.
        
        Args:
            doc_data: Parsed document data
            doc_type: Type of document ("cv" or "job")
            
        Returns:
            Processed document data with analysis
        """
        try:
            doc_id = doc_data["doc_id"]
            raw_text = doc_data["raw_text"]
            
            # Step 1: Extract sections
            sections = text_processor.extract_sections(raw_text)
            
            # Step 2: Extract semantic features
            semantic_features = nlp_processor.extract_semantic_features(raw_text)
            
            # Step 3: Generate embeddings
            embeddings = self._generate_document_embeddings(raw_text, sections)
            
            # Step 4: Analyze document content
            analysis = self._analyze_document_content(raw_text, sections, semantic_features, doc_type)
            
            # Step 5: Create processed document data
            processed_doc = {
                **doc_data,
                "processed_at": datetime.now().isoformat(),
                "sections": sections,
                "semantic_features": semantic_features,
                "embeddings": embeddings,
                "analysis": analysis
            }
            
            logger.info(f"Successfully processed {doc_type}: {doc_id}")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing {doc_type} {doc_data.get('doc_id', 'unknown')}: {str(e)}")
            return None
    
    def _generate_document_embeddings(self, raw_text: str, sections: Dict) -> Dict:
        """
        Generate embeddings for document content.
        
        Args:
            raw_text: Full document text
            sections: Extracted sections
            
        Returns:
            Dictionary containing embeddings
        """
        embeddings = {
            "full_document": nlp_processor.generate_embeddings(raw_text)[0],
            "sections": {}
        }
        
        # Generate embeddings for each section
        for section_name, section_text in sections.items():
            if section_text.strip():
                section_embedding = nlp_processor.generate_embeddings(section_text)[0]
                embeddings["sections"][section_name] = section_embedding
        
        return embeddings
    
    def _analyze_document_content(self, raw_text: str, sections: Dict, 
                                semantic_features: Dict, doc_type: str) -> Dict:
        """
        Analyze document content for insights.
        
        Args:
            raw_text: Full document text
            sections: Extracted sections
            semantic_features: Semantic features
            doc_type: Type of document ("cv" or "job")
            
        Returns:
            Analysis results
        """
        analysis = {
            "skills_analysis": {},
            "experience_analysis": {},
            "education_analysis": {},
            "overall_assessment": {}
        }
        
        # Analyze skills
        if sections.get("skills"):
            skills_analysis = self._analyze_skills_section(sections["skills"], doc_type)
            analysis["skills_analysis"] = skills_analysis
        
        # Analyze experience
        if sections.get("experience"):
            experience_analysis = self._analyze_experience_section(sections["experience"], doc_type)
            analysis["experience_analysis"] = experience_analysis
        
        # Analyze education
        if sections.get("education"):
            education_analysis = self._analyze_education_section(sections["education"], doc_type)
            analysis["education_analysis"] = education_analysis
        
        # Overall assessment
        analysis["overall_assessment"] = self._assess_overall_document(
            raw_text, sections, semantic_features, doc_type
        )
        
        return analysis
    
    def _analyze_skills_section(self, skills_text: str, doc_type: str) -> Dict:
        """
        Analyze skills section.
        
        Args:
            skills_text: Skills section text
            doc_type: Type of document ("cv" or "job")
            
        Returns:
            Skills analysis
        """
        skills = text_processor.extract_skills(skills_text)
        
        # Use transformer-based skill categorization (includes fallback)
        from utils.skill_categorizer import skill_categorizer
        
        skill_categories = skill_categorizer.categorize_skills(skills)
        
        result = {
            "extracted_skills": skills,
            "skill_categories": skill_categories,
            "total_skills": len(skills),
            "technical_skills_count": len(skills) - len(skill_categories["soft_skills"])
        }
        
        # Add document-type specific analysis
        if doc_type == "cv":
            result["has_management_experience"] = any(soft in skills_text.lower() for soft in ["leadership", "management", "team lead"])
        else:  # job
            result["requires_management"] = any(soft in skills_text.lower() for soft in ["leadership", "management", "team lead"])
        
        return result
    
    def _analyze_experience_section(self, experience_text: str, doc_type: str) -> Dict:
        """
        Analyze experience section.
        
        Args:
            experience_text: Experience section text
            doc_type: Type of document ("cv" or "job")
            
        Returns:
            Experience analysis
        """
        experience_level = text_processor.extract_experience_level(experience_text)
        
        # Extract years of experience
        import re
        years_pattern = r'(\d+)\+?\s*years?'
        years_matches = re.findall(years_pattern, experience_text.lower())
        
        if years_matches:
            years_list = [int(years) for years in years_matches]
            min_years = min(years_list)
            max_years = max(years_list)
        else:
            min_years = max_years = 0
        
        # Determine seniority level
        seniority_level = "entry"
        if max_years >= 7:
            seniority_level = "senior"
        elif max_years >= 3:
            seniority_level = "mid"
        elif max_years >= 1:
            seniority_level = "junior"
        
        result = {
            "experience_level": experience_level,
            "min_years": min_years,
            "max_years": max_years,
            "seniority_level": seniority_level
        }
        
        # Add document-type specific analysis
        if doc_type == "cv":
            result["has_management_experience"] = "lead" in experience_text.lower() or "manager" in experience_text.lower()
            result["has_team_leadership"] = "team" in experience_text.lower() and "lead" in experience_text.lower()
        else:  # job
            result["requires_management"] = "management" in experience_text.lower() or "lead" in experience_text.lower()
            result["requires_leadership"] = "leadership" in experience_text.lower() or "team lead" in experience_text.lower()
        
        return result
    
    def _analyze_education_section(self, education_text: str, doc_type: str) -> Dict:
        """
        Analyze education section.
        
        Args:
            education_text: Education section text
            doc_type: Type of document ("cv" or "job")
            
        Returns:
            Education analysis
        """
        # Extract degree information
        degrees = []
        degree_keywords = ["bachelor", "master", "phd", "doctorate", "associate", "diploma", "certification"]
        
        for keyword in degree_keywords:
            if keyword in education_text.lower():
                degrees.append(keyword.title())
        
        # Check for technical degrees
        technical_keywords = ["computer science", "engineering", "information technology", "software", "data science"]
        is_technical_degree = any(keyword in education_text.lower() for keyword in technical_keywords)
        
        result = {
            "degrees": degrees,
            "is_technical_degree": is_technical_degree,
            "has_higher_education": len(degrees) > 0
        }
        
        # Add document-type specific analysis
        if doc_type == "cv":
            result["education_level"] = "high" if "phd" in education_text.lower() or "doctorate" in education_text.lower() else "medium" if "master" in education_text.lower() else "bachelor"
        else:  # job
            result["required_degrees"] = degrees
            result["requires_technical_degree"] = is_technical_degree
            result["degree_required"] = any(required in education_text.lower() for required in ["required", "must have", "essential"])
            result["education_level"] = "high" if "phd" in education_text.lower() or "doctorate" in education_text.lower() else "medium" if "master" in education_text.lower() else "bachelor"
        
        return result
    
    def _assess_overall_document(self, raw_text: str, sections: Dict, 
                               semantic_features: Dict, doc_type: str) -> Dict:
        """
        Assess overall document quality and completeness.
        
        Args:
            raw_text: Full document text
            sections: Extracted sections
            semantic_features: Semantic features
            doc_type: Type of document ("cv" or "job")
            
        Returns:
            Overall assessment
        """
        # Calculate completeness score
        section_scores = {
            "skills": 1.0 if sections.get("skills") else 0.0,
            "experience": 1.0 if sections.get("experience") else 0.0,
            "education": 1.0 if sections.get("education") else 0.0,
            "projects": 1.0 if sections.get("projects") else 0.0
        }
        
        completeness_score = sum(section_scores.values()) / len(section_scores)
        
        # Assess content quality
        content_length = len(raw_text)
        
        result = {
            "completeness_score": completeness_score,
            "section_coverage": section_scores,
            "content_length": content_length,
            "overall_quality": "high" if completeness_score > 0.8 else "medium" if completeness_score > 0.5 else "low"
        }
        
        # Add document-type specific analysis
        if doc_type == "cv":
            result["has_contact_info"] = any(keyword in raw_text.lower() for keyword in ["email", "phone", "linkedin"])
            result["has_projects"] = bool(sections.get("projects"))
        else:  # job
            result["has_requirements"] = "required" in raw_text.lower() or "must have" in raw_text.lower()
            result["has_responsibilities"] = "responsibilities" in raw_text.lower() or "duties" in raw_text.lower()
            
            # Determine job complexity
            complexity_score = 0
            if "senior" in raw_text.lower() or "lead" in raw_text.lower():
                complexity_score = 3
            elif "mid" in raw_text.lower() or "intermediate" in raw_text.lower():
                complexity_score = 2
            else:
                complexity_score = 1
            result["complexity_level"] = complexity_score
        
        return result

# Global document processor instance
document_processor = DocumentProcessor() 