from typing import List, Dict, Optional, Tuple
import numpy as np
from loguru import logger
from config.settings import settings
from services.nlp_processor import nlp_processor
from database.vector_db import vector_db
from datetime import datetime

class SemanticProcessor:
    """Semantic processing service for CV-Job matching using embeddings and weighted scoring."""
    
    def __init__(self):
        """Initialize semantic processor."""
        self.scoring_weights = settings.SCORING_WEIGHTS
        self.matching_thresholds = settings.MATCHING_THRESHOLDS
    
    def match_opportunity_with_cvs(self, opportunity_id: str, cv_ids: List[str], n_results: int = None) -> Dict:
        """
        Match an opportunity with multiple CVs.
        
        Args:
            opportunity_id: ID of the opportunity/job
            cv_ids: List of CV IDs to match against (empty list means match against all CVs)
            n_results: Number of results to return (None means return all)
            
        Returns:
            Matching results with scores and explanations
        """
        try:
            # If cv_ids is empty, get all CV IDs from the database
            if not cv_ids:
                all_docs = vector_db.get_all_documents()
                cv_ids = [doc["doc_id"] for doc in all_docs if doc.get("doc_type") == "cv"]
                logger.info(f"Matching opportunity {opportunity_id} with all {len(cv_ids)} CVs")
            else:
                logger.info(f"Matching opportunity {opportunity_id} with {len(cv_ids)} CVs")
            
            # Get opportunity embeddings
            opportunity_embeddings = vector_db.get_document(opportunity_id)
            if opportunity_embeddings is None:
                return {"error": f"Opportunity {opportunity_id} not found"}
            
            # Get opportunity sections
            opportunity_sections = vector_db.get_document_sections(opportunity_id)
            
            # Match with each CV
            matches = []
            for cv_id in cv_ids:
                try:
                    match_result = self._match_single_cv_with_opportunity(
                        cv_id, opportunity_id, opportunity_embeddings, opportunity_sections
                    )
                    if match_result:
                        matches.append(match_result)
                except Exception as e:
                    logger.error(f"Error matching CV {cv_id}: {str(e)}")
                    continue
            
            # Sort by overall score
            matches.sort(key=lambda x: x["overall_score"], reverse=True)
            
            # Limit results if n_results is specified
            if n_results and n_results > 0:
                matches = matches[:n_results]
            
            # Add summary statistics
            if matches:
                scores = [match["overall_score"] for match in matches]
                summary = {
                    "total_matches": len(matches),
                    "total_cvs_processed": len(cv_ids),
                    "average_score": np.mean(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "excellent_matches": len([s for s in scores if s >= self.matching_thresholds["excellent_match"]]),
                    "good_matches": len([s for s in scores if s >= self.matching_thresholds["good_match"]]),
                    "fair_matches": len([s for s in scores if s >= self.matching_thresholds["fair_match"]])
                }
            else:
                summary = {
                    "total_matches": 0,
                    "total_cvs_processed": len(cv_ids),
                    "average_score": 0.0,
                    "max_score": 0.0,
                    "min_score": 0.0,
                    "excellent_matches": 0,
                    "good_matches": 0,
                    "fair_matches": 0
                }
            
            return {
                "opportunity_id": opportunity_id,
                "cv_count": len(cv_ids),
                "matches": matches,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error in opportunity matching: {str(e)}")
            return {"error": f"Error matching opportunity: {str(e)}"}
    
    def match_cv_with_opportunities(self, cv_id: str, opportunity_ids: List[str], n_results: int = None) -> Dict:
        """
        Match a CV with multiple opportunities.
        
        Args:
            cv_id: ID of the CV
            opportunity_ids: List of opportunity IDs to match against (empty list means match against all opportunities)
            n_results: Number of results to return (None means return all)
            
        Returns:
            Matching results with scores and explanations
        """
        try:
            # If opportunity_ids is empty, get all opportunity IDs from the database
            if not opportunity_ids:
                all_docs = vector_db.get_all_documents()
                opportunity_ids = [doc["doc_id"] for doc in all_docs if doc.get("doc_type") == "job"]
                logger.info(f"Matching CV {cv_id} with all {len(opportunity_ids)} opportunities")
            else:
                logger.info(f"Matching CV {cv_id} with {len(opportunity_ids)} opportunities")
            
            # Get CV embeddings
            cv_embeddings = vector_db.get_document(cv_id)
            if cv_embeddings is None:
                return {"error": f"CV {cv_id} not found"}
            
            # Get CV sections
            cv_sections = vector_db.get_document_sections(cv_id)
            
            # Match with each opportunity
            matches = []
            for opportunity_id in opportunity_ids:
                try:
                    match_result = self._match_single_cv_with_opportunity(
                        cv_id, opportunity_id, cv_embeddings, cv_sections
                    )
                    if match_result:
                        matches.append(match_result)
                except Exception as e:
                    logger.error(f"Error matching opportunity {opportunity_id}: {str(e)}")
                    continue
            
            # Sort by overall score
            matches.sort(key=lambda x: x["overall_score"], reverse=True)
            
            # Limit results if n_results is specified
            if n_results and n_results > 0:
                matches = matches[:n_results]
            
            # Add summary statistics
            if matches:
                scores = [match["overall_score"] for match in matches]
                summary = {
                    "total_matches": len(matches),
                    "total_opportunities_processed": len(opportunity_ids),
                    "average_score": np.mean(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "excellent_matches": len([s for s in scores if s >= self.matching_thresholds["excellent_match"]]),
                    "good_matches": len([s for s in scores if s >= self.matching_thresholds["good_match"]]),
                    "fair_matches": len([s for s in scores if s >= self.matching_thresholds["fair_match"]])
                }
            else:
                summary = {
                    "total_matches": 0,
                    "total_opportunities_processed": len(opportunity_ids),
                    "average_score": 0.0,
                    "max_score": 0.0,
                    "min_score": 0.0,
                    "excellent_matches": 0,
                    "good_matches": 0,
                    "fair_matches": 0
                }
            
            return {
                "cv_id": cv_id,
                "opportunity_count": len(opportunity_ids),
                "matches": matches,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error in CV matching: {str(e)}")
            return {"error": f"Error matching CV: {str(e)}"}
    
    def match_multiple_cvs_with_opportunities(self, cv_ids: List[str], opportunity_ids: List[str]) -> Dict:
        """
        Match multiple CVs with multiple opportunities.
        
        Args:
            cv_ids: List of CV IDs
            opportunity_ids: List of opportunity IDs
            
        Returns:
            Matching results with scores and explanations for all combinations
        """
        try:
            logger.info(f"Matching {len(cv_ids)} CVs with {len(opportunity_ids)} opportunities")
            
            # Validate inputs
            if not cv_ids:
                return {"error": "CV IDs list cannot be empty"}
            
            if not opportunity_ids:
                return {"error": "Opportunity IDs list cannot be empty"}
            
            # Perform matching for all combinations
            all_matches = []
            
            for cv_id in cv_ids:
                cv_matches = []
                for opportunity_id in opportunity_ids:
                    # Match single CV with single opportunity
                    match_result = self.match_opportunity_with_cvs(opportunity_id, [cv_id])
                    
                    if "error" not in match_result and "matches" in match_result and match_result["matches"]:
                        # Extract the single match result
                        single_match = match_result["matches"][0]
                        single_match["cv_id"] = cv_id
                        single_match["opportunity_id"] = opportunity_id
                        cv_matches.append(single_match)
                
                # Sort matches for this CV by score
                cv_matches.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
                all_matches.append({
                    "cv_id": cv_id,
                    "matches": cv_matches,
                    "total_matches": len(cv_matches)
                })
            
            # Calculate overall statistics
            total_combinations = len(cv_ids) * len(opportunity_ids)
            successful_matches = sum(len(cv_data["matches"]) for cv_data in all_matches)
            
            # Get best matches across all combinations
            all_single_matches = []
            for cv_data in all_matches:
                all_single_matches.extend(cv_data["matches"])
            
            all_single_matches.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
            
            return {
                "cv_count": len(cv_ids),
                "opportunity_count": len(opportunity_ids),
                "total_combinations": total_combinations,
                "successful_matches": successful_matches,
                "cv_matches": all_matches,
                "top_matches": all_single_matches[:10],  # Top 10 matches overall
                "summary": {
                    "excellent_matches": len([m for m in all_single_matches if m.get("overall_score", 0) >= 0.85]),
                    "good_matches": len([m for m in all_single_matches if 0.70 <= m.get("overall_score", 0) < 0.85]),
                    "fair_matches": len([m for m in all_single_matches if 0.50 <= m.get("overall_score", 0) < 0.70]),
                    "average_score": sum(m.get("overall_score", 0) for m in all_single_matches) / len(all_single_matches) if all_single_matches else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in multiple CVs with opportunities matching: {str(e)}")
            return {"error": f"Error matching multiple CVs with opportunities: {str(e)}"}
    
    def _match_single_cv_with_opportunity(self, cv_id: str, opportunity_id: str, 
                                        cv_embeddings: np.ndarray, cv_sections: List[str]) -> Optional[Dict]:
        """
        Match a single CV with an opportunity.
        
        Args:
            cv_id: CV ID
            opportunity_id: Opportunity ID
            cv_embeddings: CV embeddings
            cv_sections: CV sections
            
        Returns:
            Match result with scores and explanations
        """
        try:
            # Get opportunity embeddings
            opportunity_embeddings = vector_db.get_document(opportunity_id)
            if opportunity_embeddings is None:
                return None
            
            # Get opportunity sections
            opportunity_sections = vector_db.get_document_sections(opportunity_id)
            
            # Get document names (use IDs as fallback)
            cv_name = f"CV_{cv_id}"
            opportunity_name = f"Opportunity_{opportunity_id}"
            
            # Get document texts for section analysis
            from database.vector_db import vector_db
            cv_text = vector_db.get_raw_text(cv_id)
            opportunity_text = vector_db.get_raw_text(opportunity_id)
            
            if not cv_text or not opportunity_text:
                return None
            
            # Extract sections from text
            from utils.text_processing import text_processor
            cv_extracted_sections = text_processor.extract_sections(cv_text)
            opportunity_extracted_sections = text_processor.extract_sections(opportunity_text)
            
            # Calculate section-wise similarities
            section_scores = {}
            section_explanations = {}
            
            # Skills matching
            if "skills" in cv_extracted_sections and "skills" in opportunity_extracted_sections:
                skills_score, skills_explanation = self._calculate_skills_similarity(
                    cv_extracted_sections["skills"], opportunity_extracted_sections["skills"]
                )
                section_scores["skills"] = skills_score
                section_explanations["skills"] = skills_explanation
            
            # Experience matching
            if "experience" in cv_extracted_sections and "experience" in opportunity_extracted_sections:
                experience_score, experience_explanation = self._calculate_experience_similarity(
                    cv_extracted_sections["experience"], opportunity_extracted_sections["experience"]
                )
                section_scores["experience"] = experience_score
                section_explanations["experience"] = experience_explanation
            
            # Education matching
            if "education" in cv_extracted_sections and "education" in opportunity_extracted_sections:
                education_score, education_explanation = self._calculate_education_similarity(
                    cv_extracted_sections["education"], opportunity_extracted_sections["education"]
                )
                section_scores["education"] = education_score
                section_explanations["education"] = education_explanation
            
            # Projects matching
            if "projects" in cv_extracted_sections and "projects" in opportunity_extracted_sections:
                projects_score, projects_explanation = self._calculate_projects_similarity(
                    cv_extracted_sections["projects"], opportunity_extracted_sections["projects"]
                )
                section_scores["projects"] = projects_score
                section_explanations["projects"] = projects_explanation
            
            # Calculate overall score
            overall_score = self._calculate_weighted_score(section_scores)
            
            # Determine match quality
            match_quality = self._determine_match_quality(overall_score)
            
            # Generate overall explanation
            overall_explanation = self._generate_overall_explanation(
                section_scores, section_explanations, overall_score, match_quality
            )
            
            return {
                "cv_id": cv_id,
                "opportunity_id": opportunity_id,
                "cv_name": cv_name,
                "opportunity_name": opportunity_name,
                "overall_score": overall_score,
                "match_quality": match_quality,
                "section_scores": section_scores,
                "section_explanations": section_explanations,
                "overall_explanation": overall_explanation,
                "matched_sections": list(section_scores.keys())
            }
            
        except Exception as e:
            logger.error(f"Error in single CV-opportunity matching: {str(e)}")
            return None
    
    def _calculate_skills_similarity(self, cv_skills: str, job_skills: str) -> Tuple[float, str]:
        """Calculate skills similarity between CV and job."""
        try:
            # Extract skills from both documents
            from utils.text_processing import text_processor
            cv_skill_list = text_processor.extract_skills(cv_skills)
            job_skill_list = text_processor.extract_skills(job_skills)
            
            if not cv_skill_list or not job_skill_list:
                return 0.0, "No skills found in one or both documents"
            
            # Calculate skill overlap
            cv_skills_lower = [skill.lower() for skill in cv_skill_list]
            job_skills_lower = [skill.lower() for skill in job_skill_list]
            
            # Exact matches
            exact_matches = set(cv_skills_lower) & set(job_skills_lower)
            
            # Partial matches (skills that contain each other)
            partial_matches = set()
            for cv_skill in cv_skills_lower:
                for job_skill in job_skills_lower:
                    if cv_skill in job_skill or job_skill in cv_skill:
                        partial_matches.add(cv_skill)
                        partial_matches.add(job_skill)
            
            # Calculate scores
            exact_score = len(exact_matches) / max(len(cv_skills_lower), len(job_skills_lower))
            partial_score = len(partial_matches) / max(len(cv_skills_lower), len(job_skills_lower))
            
            # Weighted score (exact matches are more important)
            skills_score = (exact_score * 0.7) + (partial_score * 0.3)
            
            # Generate explanation
            if exact_matches:
                explanation = f"Strong match with {len(exact_matches)} exact skill matches: {', '.join(list(exact_matches)[:5])}"
                if len(exact_matches) > 5:
                    explanation += f" and {len(exact_matches) - 5} more"
            elif partial_matches:
                explanation = f"Partial match with {len(partial_matches)} related skills"
            else:
                explanation = "No significant skill overlap found"
            
            return min(skills_score, 1.0), explanation
            
        except Exception as e:
            logger.error(f"Error calculating skills similarity: {str(e)}")
            return 0.0, "Error calculating skills similarity"
    
    def _calculate_experience_similarity(self, cv_experience: str, job_experience: str) -> Tuple[float, str]:
        """Calculate experience similarity between CV and job."""
        try:
            # Extract experience level and years
            from utils.text_processing import text_processor
            cv_level = text_processor.extract_experience_level(cv_experience)
            job_level = text_processor.extract_experience_level(job_experience)
            
            # Extract years of experience
            import re
            cv_years = re.findall(r'(\d+)\+?\s*years?', cv_experience.lower())
            job_years = re.findall(r'(\d+)\+?\s*years?', job_experience.lower())
            
            cv_max_years = max([int(y) for y in cv_years]) if cv_years else 0
            job_min_years = min([int(y) for y in job_years]) if job_years else 0
            
            # Calculate experience score
            if job_min_years == 0:
                experience_score = 0.5  # Neutral if no specific requirement
            elif cv_max_years >= job_min_years:
                experience_score = 1.0  # Meets or exceeds requirement
            else:
                # Partial score based on how close they are
                experience_score = max(0.0, cv_max_years / job_min_years)
            
            # Generate explanation
            if cv_max_years >= job_min_years:
                explanation = f"Experience requirement met: {cv_max_years} years vs {job_min_years} required"
            elif cv_max_years > 0:
                explanation = f"Experience gap: {cv_max_years} years vs {job_min_years} required"
            else:
                explanation = "No clear experience information found"
            
            return experience_score, explanation
            
        except Exception as e:
            logger.error(f"Error calculating experience similarity: {str(e)}")
            return 0.0, "Error calculating experience similarity"
    
    def _calculate_education_similarity(self, cv_education: str, job_education: str) -> Tuple[float, str]:
        """Calculate education similarity between CV and job."""
        try:
            # Extract degree information
            degree_keywords = ["bachelor", "master", "phd", "doctorate", "associate", "diploma"]
            
            cv_degrees = [keyword for keyword in degree_keywords if keyword in cv_education.lower()]
            job_degrees = [keyword for keyword in degree_keywords if keyword in job_education.lower()]
            
            # Check for technical degree requirements
            technical_keywords = ["computer science", "engineering", "information technology", "software", "data science"]
            cv_technical = any(keyword in cv_education.lower() for keyword in technical_keywords)
            job_technical = any(keyword in job_education.lower() for keyword in technical_keywords)
            
            # Calculate education score
            degree_score = 0.0
            if job_degrees:
                if cv_degrees:
                    # Check if CV meets or exceeds job requirements
                    cv_highest = self._get_degree_level(cv_degrees)
                    job_highest = self._get_degree_level(job_degrees)
                    if cv_highest >= job_highest:
                        degree_score = 1.0
                    else:
                        degree_score = 0.5
                else:
                    degree_score = 0.0
            else:
                degree_score = 0.5  # No specific requirement
            
            # Technical degree bonus
            technical_score = 1.0 if (not job_technical or cv_technical) else 0.0
            
            # Combined score
            education_score = (degree_score * 0.7) + (technical_score * 0.3)
            
            # Generate explanation
            if cv_degrees and job_degrees:
                explanation = f"Education match: {', '.join(cv_degrees)} vs {', '.join(job_degrees)} required"
            elif cv_degrees:
                explanation = f"Has {', '.join(cv_degrees)} degree"
            elif job_degrees:
                explanation = f"Education requirement: {', '.join(job_degrees)}"
            else:
                explanation = "No specific education requirements"
            
            return education_score, explanation
            
        except Exception as e:
            logger.error(f"Error calculating education similarity: {str(e)}")
            return 0.0, "Error calculating education similarity"
    
    def _calculate_projects_similarity(self, cv_projects: str, job_projects: str) -> Tuple[float, str]:
        """Calculate projects similarity between CV and job."""
        try:
            # Simple similarity based on project-related keywords
            project_keywords = ["project", "portfolio", "achievement", "accomplishment", "deliverable"]
            
            cv_project_count = sum(1 for keyword in project_keywords if keyword in cv_projects.lower())
            job_project_count = sum(1 for keyword in project_keywords if keyword in job_projects.lower())
            
            # Calculate score based on project presence
            if cv_project_count > 0 and job_project_count > 0:
                projects_score = min(1.0, cv_project_count / max(job_project_count, 1))
                explanation = f"Project experience: {cv_project_count} project indicators found"
            elif cv_project_count > 0:
                projects_score = 0.8
                explanation = "Has project experience"
            else:
                projects_score = 0.3
                explanation = "Limited project information"
            
            return projects_score, explanation
            
        except Exception as e:
            logger.error(f"Error calculating projects similarity: {str(e)}")
            return 0.0, "Error calculating projects similarity"
    
    def _get_degree_level(self, degrees: List[str]) -> int:
        """Get the highest degree level."""
        level_mapping = {
            "associate": 1,
            "diploma": 1,
            "bachelor": 2,
            "master": 3,
            "phd": 4,
            "doctorate": 4
        }
        
        levels = [level_mapping.get(degree, 0) for degree in degrees]
        return max(levels) if levels else 0
    
    def _calculate_weighted_score(self, section_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        total_score = 0.0
        total_weight = 0.0
        
        for section, score in section_scores.items():
            weight = self.scoring_weights.get(f"{section}_match", 0.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_match_quality(self, score: float) -> str:
        """Determine match quality based on score."""
        if score >= self.matching_thresholds["excellent_match"]:
            return "excellent"
        elif score >= self.matching_thresholds["good_match"]:
            return "good"
        elif score >= self.matching_thresholds["fair_match"]:
            return "fair"
        elif score >= self.matching_thresholds["minimum_match"]:
            return "poor"
        else:
            return "unmatched"
    
    def _generate_overall_explanation(self, section_scores: Dict[str, float], 
                                    section_explanations: Dict[str, str], 
                                    overall_score: float, match_quality: str) -> str:
        """Generate overall explanation for the match."""
        explanations = []
        
        # Add section explanations
        for section, explanation in section_explanations.items():
            score = section_scores.get(section, 0.0)
            if score > 0.5:
                explanations.append(f"{section.title()}: {explanation}")
        
        # Add overall assessment
        quality_descriptions = {
            "excellent": "This is an excellent match with strong alignment across all areas.",
            "good": "This is a good match with solid alignment in key areas.",
            "fair": "This is a fair match with some alignment but areas for improvement.",
            "poor": "This is a poor match with limited alignment.",
            "unmatched": "This match does not meet minimum requirements."
        }
        
        overall_desc = quality_descriptions.get(match_quality, "Match quality unclear.")
        
        if explanations:
            return f"{overall_desc} Key strengths: {'; '.join(explanations[:3])}"
        else:
            return overall_desc
    
    def get_semantic_insights(self, cv_id: str, job_id: str) -> Dict:
        """
        Get detailed semantic insights between a CV and job.
        
        Args:
            cv_id: CV ID
            job_id: Job ID
            
        Returns:
            Detailed semantic insights
        """
        try:
            # Get document texts
            cv_text = vector_db.get_raw_text(cv_id)
            job_text = vector_db.get_raw_text(job_id)
            
            if not cv_text or not job_text:
                return {"error": "One or both documents not found"}
            
            # Get document names (use IDs as fallback)
            cv_name = f"CV_{cv_id}"
            job_name = f"Job_{job_id}"
            
            # Perform semantic analysis
            insights = nlp_processor.understand_semantic_relationships(cv_text, job_text)
            
            # Add document information
            insights.update({
                "cv_id": cv_id,
                "job_id": job_id,
                "cv_name": cv_name,
                "job_name": job_name,
                "analysis_timestamp": datetime.now().isoformat()
            })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting semantic insights: {str(e)}")
            return {"error": f"Error getting semantic insights: {str(e)}"}
    
    def search_cvs_by_query(self, query: str, n_results: int = 10) -> List[str]:
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
            
            # Use vector database to search for CVs
            cv_ids = vector_db.search_similar_ids(
                query_text=query,
                doc_type="cv",
                n_results=n_results
            )
            
            logger.info(f"Found {len(cv_ids)} CVs matching query: {query}")
            return cv_ids
            
        except Exception as e:
            logger.error(f"Error searching CVs by query: {str(e)}")
            return []
    
    def search_opportunities_by_query(self, query: str, n_results: int = 10) -> List[str]:
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
            
            # Use vector database to search for opportunities
            opportunity_ids = vector_db.search_similar_ids(
                query_text=query,
                doc_type="job",
                n_results=n_results
            )
            
            logger.info(f"Found {len(opportunity_ids)} opportunities matching query: {query}")
            return opportunity_ids
            
        except Exception as e:
            logger.error(f"Error searching opportunities by query: {str(e)}")
            return []

# Global semantic processor instance
semantic_processor = SemanticProcessor() 