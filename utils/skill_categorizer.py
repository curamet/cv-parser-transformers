from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional
import torch
import numpy as np
from loguru import logger
import json
import re

class SkillCategorizer:
    """Transformer-based skill categorization using zero-shot classification."""
    
    def __init__(self):
        """Initialize the skill categorizer with transformer models."""
        self.zero_shot_classifier = None
        self.skill_embeddings = None
        self.category_embeddings = None
        self._model_loaded = False
        self._initialize_categories()
    
    def _load_models(self):
        """Load transformer models for skill categorization (lazy loading)."""
        if self._model_loaded:
            return
            
        try:
            logger.info("Loading zero-shot classification model (this may take a few seconds)...")
            # Load zero-shot classification pipeline
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            self._model_loaded = True
            logger.info("✅ Zero-shot classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading zero-shot classifier: {e}")
            self.zero_shot_classifier = None
    
    def _initialize_categories(self):
        """Initialize skill categories with their descriptions."""
        self.categories = {
            "programming_languages": [
                "programming language", "coding language", "development language",
                "scripting language", "programming technology"
            ],
            "frameworks": [
                "framework", "library", "development framework", "web framework",
                "application framework", "software framework"
            ],
            "databases": [
                "database", "database system", "data storage", "database technology",
                "data management system"
            ],
            "cloud_platforms": [
                "cloud platform", "cloud service", "infrastructure", "deployment platform",
                "cloud computing", "containerization", "orchestration"
            ],
            "tools": [
                "development tool", "software tool", "utility", "development utility",
                "programming tool", "development environment"
            ],
            "soft_skills": [
                "soft skill", "interpersonal skill", "communication skill", "leadership skill",
                "personal skill", "behavioral skill"
            ]
        }
    
    def categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Categorize skills using transformer-based zero-shot classification.
        
        Args:
            skills: List of skills to categorize
            
        Returns:
            Dictionary with categorized skills
        """
        if not skills:
            return self._get_empty_categories()
        
        try:
            # Load model if not already loaded
            if not self._model_loaded:
                self._load_models()
            
            # Use zero-shot classification if available
            if self.zero_shot_classifier:
                return self._categorize_with_zero_shot(skills)
            else:
                # Fallback to rule-based categorization
                return self._categorize_with_rules(skills)
                
        except Exception as e:
            logger.error(f"Error in skill categorization: {e}")
            return self._categorize_with_rules(skills)
    
    def _categorize_with_zero_shot(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Categorize skills using zero-shot classification.
        
        Args:
            skills: List of skills to categorize
            
        Returns:
            Dictionary with categorized skills
        """
        # Prepare candidate labels for zero-shot classification
        candidate_labels = []
        for category, descriptions in self.categories.items():
            candidate_labels.extend(descriptions)
        
        # Add some specific examples for better classification
        specific_examples = [
            "Python programming language", "JavaScript programming language",
            "React framework", "Django framework", "MySQL database", "PostgreSQL database",
            "AWS cloud platform", "Docker containerization", "Git development tool",
            "Leadership soft skill", "Communication soft skill"
        ]
        candidate_labels.extend(specific_examples)
        
        categorized_skills = self._get_empty_categories()
        
        # Process skills in batches for efficiency
        batch_size = 10
        for i in range(0, len(skills), batch_size):
            batch_skills = skills[i:i + batch_size]
            
            for skill in batch_skills:
                try:
                    # Classify the skill
                    if self.zero_shot_classifier is not None:
                        result = self.zero_shot_classifier(
                            skill,
                            candidate_labels,
                            hypothesis_template="This is a {}."
                        )
                        
                        # Get the best matching category
                        if isinstance(result, dict) and 'labels' in result and 'scores' in result:
                            best_label = result['labels'][0]
                            confidence = result['scores'][0]
                            
                            # Map the label back to a category
                            category = self._map_label_to_category(best_label, confidence)
                            categorized_skills[category].append(skill)
                        else:
                            # Fallback to tools if result format is unexpected
                            categorized_skills["tools"].append(skill)
                    else:
                        # Fallback to tools if classifier is not available
                        categorized_skills["tools"].append(skill)
                    
                except Exception as e:
                    logger.warning(f"Failed to classify skill '{skill}': {e}")
                    # Put in tools as default
                    categorized_skills["tools"].append(skill)
        
        return categorized_skills
    
    def _map_label_to_category(self, label: str, confidence: float) -> str:
        """
        Map a classification label to a skill category.
        
        Args:
            label: The classified label
            confidence: Classification confidence score
            
        Returns:
            Skill category
        """
        # If confidence is too low, default to tools
        if confidence < 0.3:
            return "tools"
        
        # Map specific examples
        specific_mappings = {
            "python programming language": "programming_languages",
            "javascript programming language": "programming_languages",
            "react framework": "frameworks",
            "django framework": "frameworks",
            "mysql database": "databases",
            "postgresql database": "databases",
            "aws cloud platform": "cloud_platforms",
            "docker containerization": "cloud_platforms",
            "git development tool": "tools",
            "leadership soft skill": "soft_skills",
            "communication soft skill": "soft_skills"
        }
        
        if label.lower() in specific_mappings:
            return specific_mappings[label.lower()]
        
        # Map based on category descriptions
        for category, descriptions in self.categories.items():
            if any(desc in label.lower() for desc in descriptions):
                return category
        
        # Default to tools
        return "tools"
    
    def _categorize_with_rules(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Fallback rule-based skill categorization.
        
        Args:
            skills: List of skills to categorize
            
        Returns:
            Dictionary with categorized skills
        """
        categorized_skills = self._get_empty_categories()
        
        # Enhanced skill dictionaries
        skill_patterns = {
            "programming_languages": [
                r"\b(python|java|javascript|typescript|c\+\+|c#|go|rust|php|ruby|swift|kotlin|scala|r|matlab|perl|bash|shell|powershell|vba|cobol|fortran|pascal|ada|lisp|haskell|erlang|elixir|clojure|groovy|dart|nim|zig|crystal|julia|d|f#|ocaml|racket)\b"
            ],
            "frameworks": [
                r"\b(react|angular|vue|svelte|django|flask|fastapi|spring|express|laravel|rails|asp\.net|node\.js|jquery|bootstrap|tailwind|material-ui|next\.js|nuxt\.js|gatsby|ember|backbone|meteor|sails|hapi|koa|fastify|nest\.js|strapi|wordpress|drupal|joomla|magento|shopify|tensorflow|pytorch|scikit-learn|pandas|numpy|matplotlib|seaborn|plotly|d3\.js|chart\.js|three\.js|unity|unreal|godot)\b"
            ],
            "databases": [
                r"\b(mysql|postgresql|mongodb|redis|elasticsearch|oracle|sql server|sqlite|mariadb|cassandra|dynamodb|firebase|supabase|couchdb|neo4j|influxdb|timescaledb|clickhouse|snowflake|bigquery|redshift|hbase|hive|impala|presto|kafka|rabbitmq|apache spark|hadoop)\b"
            ],
            "cloud_platforms": [
                r"\b(aws|azure|gcp|docker|kubernetes|terraform|ansible|jenkins|gitlab|github|bitbucket|circleci|travis|teamcity|bamboo|vault|consul|nomad|istio|linkerd|prometheus|grafana|elk|splunk|datadog|newrelic|cloudflare|heroku|vercel|netlify|digitalocean|linode|vultr|ovh|alibaba cloud|ibm cloud)\b"
            ],
            "soft_skills": [
                r"\b(leadership|communication|teamwork|problem solving|project management|collaboration|mentoring|coaching|presentation|negotiation|conflict resolution|time management|organization|adaptability|creativity|critical thinking|analytical thinking|strategic thinking|decision making|emotional intelligence|empathy|patience|flexibility|initiative|self-motivation|attention to detail|multitasking|prioritization|delegation|feedback|active listening)\b"
            ]
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            categorized = False
            
            # Check each category
            for category, patterns in skill_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, skill_lower):
                        categorized_skills[category].append(skill)
                        categorized = True
                        break
                if categorized:
                    break
            
            # If not categorized, put in tools
            if not categorized:
                categorized_skills["tools"].append(skill)
        
        return categorized_skills
    
    def _get_empty_categories(self) -> Dict[str, List[str]]:
        """Get empty category structure."""
        return {
            "programming_languages": [],
            "frameworks": [],
            "databases": [],
            "cloud_platforms": [],
            "tools": [],
            "soft_skills": []
        }
    
    def get_skill_insights(self, skills: List[str]) -> Dict:
        """
        Get insights about the skills.
        
        Args:
            skills: List of skills
            
        Returns:
            Dictionary with skill insights
        """
        categorized = self.categorize_skills(skills)
        
        insights = {
            "total_skills": len(skills),
            "categories": categorized,
            "category_counts": {cat: len(skills_list) for cat, skills_list in categorized.items()},
            "primary_category": max(categorized.items(), key=lambda x: len(x[1]))[0] if skills else "tools",
            "technical_skills_count": len(skills) - len(categorized["soft_skills"]),
            "soft_skills_count": len(categorized["soft_skills"]),
            "diversity_score": self._calculate_diversity_score(categorized)
        }
        
        return insights
    
    def _calculate_diversity_score(self, categorized: Dict[str, List[str]]) -> float:
        """
        Calculate skill diversity score.
        
        Args:
            categorized: Categorized skills
            
        Returns:
            Diversity score between 0 and 1
        """
        category_counts = [len(skills) for skills in categorized.values()]
        total_skills = sum(category_counts)
        
        if total_skills == 0:
            return 0.0
        
        # Calculate entropy-based diversity
        proportions = [count / total_skills for count in category_counts if count > 0]
        entropy = -sum(p * np.log2(p) for p in proportions)
        max_entropy = np.log2(len(proportions))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0

# Global skill categorizer instance
skill_categorizer = SkillCategorizer() 