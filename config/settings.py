import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings:

    """LLM Configuration"""
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Options: "ollama", "openai", "anthropic"
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")  # Model name
    LLM_API_KEY = os.getenv("LLM_API_KEY", None)  # For online providers like OpenAI

    # CV Analysis prompt
    CV_ANALYSIS_PROMPT = """
    You are an expert technical recruiter and career coach with 15+ years of experience evaluating CVs for technology roles. Your task is to provide a comprehensive analysis of the provided CV using the structured format below.

    ## Analysis Framework

    Please analyze the CV across these key dimensions and provide scores out of 10 for each category:

    ### 1. Technical Skills Match (Score: X/10)
    - Evaluate relevance of technical skills to the target role
    - Assess depth vs. breadth of technical expertise
    - Consider modern vs. legacy technology stack
    - Comment on skill progression and learning curve

    ### 2. Experience Relevance (Score: X/10)
    - Analyze how well previous roles align with target position
    - Evaluate career progression and growth trajectory
    - Assess industry experience and domain knowledge
    - Consider role complexity and responsibility level

    ### 3. Certifications (Score: X/10)
    - Review current certifications and their relevance
    - Evaluate certification recency and vendor reputation
    - Identify missing certifications that would strengthen profile
    - Consider certification-to-experience ratio

    ### 4. Soft Skills & Language (Score: X/10)
    - Assess communication skills demonstration
    - Evaluate leadership and collaboration evidence
    - Review problem-solving and analytical thinking examples
    - Consider cultural fit and interpersonal skills

    ### 5. Quantifiable Achievements (Score: X/10)
    - Analyze use of metrics and measurable outcomes
    - Evaluate business impact demonstration
    - Assess specificity and credibility of achievements
    - Consider scale and complexity of accomplishments

    ### 6. Tools & Ecosystem Use (Score: X/10)
    - Review breadth of tools and platforms used
    - Evaluate tool selection appropriateness
    - Assess integration and ecosystem understanding
    - Consider innovation and adoption of emerging tools

    ## Output Format

    Present your analysis in this exact structure:

    **CV Strength Evaluation**

    **Technical Skills Match: X/10**
    [Detailed assessment and comments]

    **Experience Relevance: X/10**
    [Detailed assessment and comments]

    **Certifications: X/10**
    [Detailed assessment and comments]

    **Soft Skills & Language: X/10**
    [Detailed assessment and comments]

    **Quantifiable Achievements: X/10**
    [Detailed assessment and comments]

    **Tools & Ecosystem Use: X/10**
    [Detailed assessment and comments]

    **📈 Estimated Interview Call Probability**
    ~X% for [specific role type] roles with [key focus areas], particularly in organizations using [relevant technologies/methodologies].

    **🔧 Suggested Improvements**
    1. **[Improvement Area 1]**
       [Specific actionable recommendation]

    2. **[Improvement Area 2]**
       [Specific actionable recommendation]

    3. **[Improvement Area 3]**
       [Specific actionable recommendation]

    4. **[Improvement Area 4]**
       [Specific actionable recommendation]

    5. **[Improvement Area 5]**
       [Specific actionable recommendation]

    6. **[Improvement Area 6]**
       [Specific actionable recommendation]

    **✅ Summary**
    [2-3 sentence overall assessment of the CV's strengths and competitive positioning, including geographic market considerations if relevant]

    ## Analysis Guidelines

    - Be specific and actionable in your feedback
    - Consider current market trends and employer expectations
    - Provide concrete examples for improvements
    - Balance constructive criticism with positive reinforcement
    - Consider the target role level (junior, mid-level, senior, lead)
    - Include industry-specific insights where applicable
    - Suggest realistic timelines for implementing improvements
    - Consider regional market differences if mentioned

    ## Context Requirements

    Before analyzing, please specify:
    - Target role/position type
    - Target company size/industry (if known)
    - Geographic market focus
    - Career level (junior/mid/senior/executive)

    Now please analyze the following CV:
    """

    """Configuration settings for the semantic CV-Job matching system."""
    
    # Model configurations
    EMBEDDING_MODELS = {
        "primary": "sentence-transformers/all-mpnet-base-v2"
    }
    
    # Default model to use
    DEFAULT_MODEL = "primary"
    
    # Scoring weights for different sections
    SCORING_WEIGHTS = {
        "skills_match": 0.40,
        "experience_match": 0.35,
        "education_match": 0.15,
        "projects_match": 0.10
    }
    
    # Vector database settings
    VECTOR_DB_SETTINGS = {
        "database_type": os.getenv("VECTOR_DB_TYPE", "chromadb"),  # Options: "chromadb", "pinecone", "qdrant"
        "collection_name": os.getenv("VECTOR_DB_COLLECTION_NAME", "cv_job_embeddings"),
        "embedding_dimension": 768,  # For all-mpnet-base-v2
        "distance_metric": "cosine",
        "persist_directory": os.getenv("VECTOR_DB_PERSIST_DIRECTORY", "./vector_db"),
        
        # ChromaDB specific settings
        "chromadb_collection_name": os.getenv("CHROMADB_COLLECTION_NAME", "cv_job_embeddings"),
        "chromadb_persist_directory": os.getenv("CHROMADB_PERSIST_DIRECTORY", "./vector_db"),
        
        # Pinecone specific settings
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_environment": os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp"),
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME", "cv-job-matcher"),
        
        # Qdrant specific settings
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "qdrant_collection_name": os.getenv("QDRANT_COLLECTION_NAME", "cv-job-matcher")
    }
    
    # Processing settings
    PROCESSING_SETTINGS = {
        "batch_size": int(os.getenv("BATCH_SIZE", "32")),
        "max_text_length": int(os.getenv("MAX_TEXT_LENGTH", "512")),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "200")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50"))
    }
    
    # File processing settings
    FILE_SETTINGS = {
        "supported_formats": [".pdf", ".docx", ".txt"],
        "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "10")),
        "upload_directory": os.getenv("UPLOAD_DIRECTORY", "./uploads")
    }
    
    # API settings
    API_SETTINGS = {
        "host": os.getenv("API_HOST", "0.0.0.0"),
        "port": int(os.getenv("API_PORT", "8000")),
        "debug": os.getenv("API_DEBUG", "true").lower() == "true",
        "workers": int(os.getenv("API_WORKERS", "1"))
    }
    
    # Logging settings
    LOGGING_SETTINGS = {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        "file": os.getenv("LOG_FILE", "logs/semantic_matcher.log")
    }
    
    # Semantic matching thresholds
    MATCHING_THRESHOLDS = {
        "excellent_match": float(os.getenv("EXCELLENT_MATCH_THRESHOLD", "0.85")),
        "good_match": float(os.getenv("GOOD_MATCH_THRESHOLD", "0.70")),
        "fair_match": float(os.getenv("FAIR_MATCH_THRESHOLD", "0.50")),
        "minimum_match": float(os.getenv("MINIMUM_MATCH_THRESHOLD", "0.30"))
    }
    
    # Section detection keywords
    SECTION_KEYWORDS = {
        "skills": ["skills", "technologies", "programming languages", "tools", "frameworks"],
        "experience": ["experience", "work history", "employment", "career", "professional"],
        "education": ["education", "academic", "degree", "university", "college", "certification"],
        "projects": ["projects", "portfolio", "achievements", "accomplishments"]
    }
    
    @classmethod
    def get_model_name(cls, model_key: str = None) -> str:
        """Get the model name for the specified key."""
        if model_key is None:
            model_key = cls.DEFAULT_MODEL
        return cls.EMBEDDING_MODELS.get(model_key, cls.EMBEDDING_MODELS["primary"])
    
    @classmethod
    def get_embedding_dimension(cls, model_name: str) -> int:
        """Get embedding dimension for the specified model."""
        dimensions = {
            "sentence-transformers/all-mpnet-base-v2": 768
        }
        return dimensions.get(model_name, 768)

# Global settings instance
settings = Settings() 