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
        logger.info(f"Step 1: Parsing {doc_type} documents...")
        if doc_type == "cv":
            parsed_docs = document_parser.process_multiple_cvs(files)
        else:
            parsed_docs = document_parser.process_multiple_jobs(files)
            
        logger.info(f"Parsing completed. Got {len(parsed_docs) if parsed_docs else 0} parsed documents")
        
        if not parsed_docs:
            logger.error(f"No {doc_type}s were successfully parsed")
            return []
        
        # Step 2: Process each document
        logger.info(f"Step 2: Processing {len(parsed_docs)} documents...")
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
                
                logger.info(f"Processing document {i+1}/{len(parsed_docs)} with ID: {doc_data.get('doc_id', 'unknown')}")
                processed_doc = self._process_single_document(doc_data, doc_type)
                if processed_doc:
                    processed_docs.append(processed_doc)
                    logger.info(f"Successfully processed document {i+1}")
                else:
                    logger.error(f"Failed to process document {i+1}")
            except Exception as e:
                logger.error(f"Error processing {doc_type} {doc_data.get('doc_id', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Document processing completed. Got {len(processed_docs)} processed documents")
        
        # Step 3: Store vectors
        if processed_docs:
            logger.info(f"Step 3: Storing {len(processed_docs)} documents in vector database...")
            stored_ids = vector_db.store_documents(processed_docs)
            logger.info(f"Vector storage completed. Stored {len(stored_ids)} {doc_type} vectors in database")
        else:
            logger.warning("No documents to store in vector database")
        
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
            
            logger.info(f"Starting processing for {doc_type} {doc_id} (text length: {len(raw_text)})")
            
            # Step 1: Extract sections
            logger.info(f"Step 1: Extracting sections for {doc_id}...")
            start_time = datetime.now()
            sections = text_processor.extract_sections(raw_text)
            section_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Sections extracted in {section_time:.2f}s. Found sections: {list(sections.keys())}")
            
            # Step 2: Extract semantic features
            logger.info(f"Step 2: Extracting semantic features for {doc_id}...")
            start_time = datetime.now()
            semantic_features = nlp_processor.extract_semantic_features(raw_text)
            semantic_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Semantic features extracted in {semantic_time:.2f}s")
            
            # Step 3: Generate embeddings
            logger.info(f"Step 3: Generating embeddings for {doc_id}...")
            start_time = datetime.now()
            embeddings = self._generate_document_embeddings(raw_text, sections)
            embedding_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Embeddings generated in {embedding_time:.2f}s. Generated {len(embeddings['sections'])} section embeddings")
            
            # Step 4: Analyze document content
            logger.info(f"Step 4: Analyzing document content for {doc_id}...")
            start_time = datetime.now()
            analysis = self._analyze_document_content(raw_text, sections, semantic_features, doc_type)
            analysis_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Document analysis completed in {analysis_time:.2f}s")
            
            # Step 5: Create processed document data
            logger.info(f"Step 5: Creating final processed document for {doc_id}...")
            processed_doc = {
                **doc_data,
                "processed_at": datetime.now().isoformat(),
                "sections": sections,
                "semantic_features": semantic_features,
                "embeddings": embeddings,
                "analysis": analysis
            }
            
            total_time = section_time + semantic_time + embedding_time + analysis_time
            logger.info(f"Successfully processed {doc_type} {doc_id} in {total_time:.2f}s total")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing {doc_type} {doc_data.get('doc_id', 'unknown')}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        logger.info(f"Starting embedding generation for document (text length: {len(raw_text)})")
        
        # Generate full document embedding
        logger.info("Generating full document embedding...")
        start_time = datetime.now()
        full_doc_embedding = nlp_processor.generate_embeddings(raw_text)[0]
        full_doc_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Full document embedding generated in {full_doc_time:.2f}s")
        
        embeddings = {
            "full_document": full_doc_embedding,
            "sections": {}
        }
        
        # Generate embeddings for each section
        logger.info(f"Generating section embeddings for {len(sections)} sections...")
        section_embeddings_count = 0
        for section_name, section_text in sections.items():
            if section_text.strip():
                logger.info(f"Generating embedding for section: {section_name} (length: {len(section_text)})")
                start_time = datetime.now()
                section_embedding = nlp_processor.generate_embeddings(section_text)[0]
                section_time = (datetime.now() - start_time).total_seconds()
                embeddings["sections"][section_name] = section_embedding
                section_embeddings_count += 1
                logger.info(f"Section '{section_name}' embedding generated in {section_time:.2f}s")
        
        total_embeddings = 1 + section_embeddings_count  # full doc + sections
        logger.info(f"Embedding generation completed. Generated {total_embeddings} embeddings total")
        
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
        try:
            # Extract skills using text processing (faster approach)
            from utils.text_processing import text_processor
            skills = text_processor.extract_skills(skills_text)
            
            # Use simple rule-based categorization instead of slow zero-shot classification
            skill_categories = self._categorize_skills_simple(skills)
            
            return {
                "extracted_skills": skills,
                "skill_categories": skill_categories,
                "total_skills": len(skills),
                "technical_skills_count": len([s for s in skills if any(cat in skill_categories for cat in ["programming_languages", "frameworks", "databases", "cloud_platforms", "tools"])]),
                "has_management_experience": any("management" in skill.lower() or "lead" in skill.lower() for skill in skills)
            }
        except Exception as e:
            logger.error(f"Error analyzing skills: {str(e)}")
            return {
                "extracted_skills": [],
                "skill_categories": {},
                "total_skills": 0,
                "technical_skills_count": 0,
                "has_management_experience": False
            }
    
    def _categorize_skills_simple(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Simple rule-based skill categorization (faster than zero-shot classification).
        
        Args:
            skills: List of skills to categorize
            
        Returns:
            Dictionary with categorized skills
        """
        categories = {
            "programming_languages": [],
            "frameworks": [],
            "databases": [],
            "cloud_platforms": [],
            "tools": [],
            "soft_skills": []
        }
        
        # Simple keyword-based categorization
        for skill in skills:
            skill_lower = skill.lower()
            
            # Programming languages
            if any(lang in skill_lower for lang in ["python", "java", "javascript", "typescript", "c++", "c#", "php", "ruby", "go", "rust", "swift", "kotlin", "scala"]):
                categories["programming_languages"].append(skill)
            # Frameworks
            elif any(fw in skill_lower for fw in ["react", "angular", "vue", "node", "express", "django", "flask", "spring", "laravel", "rails", "asp.net", "dotnet"]):
                categories["frameworks"].append(skill)
            # Databases
            elif any(db in skill_lower for db in ["mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "sql server", "elasticsearch"]):
                categories["databases"].append(skill)
            # Cloud platforms
            elif any(cloud in skill_lower for cloud in ["aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform", "jenkins", "gitlab", "github"]):
                categories["cloud_platforms"].append(skill)
            # Tools
            elif any(tool in skill_lower for tool in ["git", "jira", "confluence", "postman", "swagger", "figma", "photoshop", "excel", "word", "powerpoint"]):
                categories["tools"].append(skill)
            # Soft skills
            elif any(soft in skill_lower for soft in ["leadership", "communication", "teamwork", "problem solving", "analytical", "creative", "organized", "detail-oriented"]):
                categories["soft_skills"].append(skill)
            else:
                # Default to tools if no specific category matches
                categories["tools"].append(skill)
        
        return categories
    
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