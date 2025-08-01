import re
import nltk
from typing import List, Dict, Tuple, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from loguru import logger
# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextProcessor:
    """Text processing utilities for CV and job description analysis."""
    
    def __init__(self):
        """Initialize text processor with NLP models."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        # Remove multiple periods
        text = re.sub(r'\.+', '.', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Automatically detect and extract sections from CV or job description.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary with section names and their content
        """
        sections = {
            "skills": "",
            "experience": "",
            "education": "",
            "projects": "",
            "summary": ""
        }
        
        # Use spaCy for better section detection
        doc = self.nlp(text)
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_section = "summary"
        section_content = []
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Detect section based on keywords and context
            if self._is_skills_section(paragraph_lower):
                if section_content:
                    sections[current_section] = '\n\n'.join(section_content)
                current_section = "skills"
                section_content = [paragraph]
            elif self._is_experience_section(paragraph_lower):
                if section_content:
                    sections[current_section] = '\n\n'.join(section_content)
                current_section = "experience"
                section_content = [paragraph]
            elif self._is_education_section(paragraph_lower):
                if section_content:
                    sections[current_section] = '\n\n'.join(section_content)
                current_section = "education"
                section_content = [paragraph]
            elif self._is_projects_section(paragraph_lower):
                if section_content:
                    sections[current_section] = '\n\n'.join(section_content)
                current_section = "projects"
                section_content = [paragraph]
            else:
                section_content.append(paragraph)
        
        # Add the last section
        if section_content:
            sections[current_section] = '\n\n'.join(section_content)
        
        return sections
    
    def _is_skills_section(self, text: str) -> bool:
        """Check if text represents a skills section."""
        skills_keywords = [
            "skills", "technologies", "programming languages", "tools", 
            "frameworks", "languages", "technologies", "competencies",
            "technical skills", "proficiencies"
        ]
        return any(keyword in text for keyword in skills_keywords)
    
    def _is_experience_section(self, text: str) -> bool:
        """Check if text represents an experience section."""
        experience_keywords = [
            "experience", "work history", "employment", "career", 
            "professional experience", "work experience", "employment history",
            "positions held", "roles"
        ]
        return any(keyword in text for keyword in experience_keywords)
    
    def _is_education_section(self, text: str) -> bool:
        """Check if text represents an education section."""
        education_keywords = [
            "education", "academic", "degree", "university", "college", 
            "certification", "qualifications", "academic background",
            "degrees", "certifications"
        ]
        return any(keyword in text for keyword in education_keywords)
    
    def _is_projects_section(self, text: str) -> bool:
        """Check if text represents a projects section."""
        projects_keywords = [
            "projects", "portfolio", "achievements", "accomplishments",
            "key projects", "notable projects", "work samples"
        ]
        return any(keyword in text for keyword in projects_keywords)
    
    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using NLP techniques.
        
        Args:
            text: Text containing skills information
            
        Returns:
            List of extracted skills
        """
        skills = []
        
        # Use spaCy for named entity recognition
        doc = self.nlp(text)
        
        # Extract technical terms and proper nouns
        for token in doc:
            # Look for technical terms, proper nouns, and compound nouns
            if (token.pos_ in ['PROPN', 'NOUN'] and 
                len(token.text) > 2 and 
                token.text.lower() not in self.stop_words):
                skills.append(token.text)
        
        # Extract compound technical terms
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3:
                skills.append(chunk.text)
        
        # Remove duplicates and clean
        skills = list(set([skill.strip() for skill in skills if skill.strip()]))
        
        return skills
    
    def extract_experience_level(self, text: str) -> Optional[str]:
        """
        Extract experience level from text.
        
        Args:
            text: Text containing experience information
            
        Returns:
            Experience level (junior, mid, senior, lead, etc.)
        """
        text_lower = text.lower()
        
        # Define experience level patterns
        experience_patterns = {
            "junior": ["junior", "entry level", "0-2 years", "1-2 years", "beginner"],
            "mid": ["mid level", "mid-level", "3-5 years", "4-6 years", "intermediate"],
            "senior": ["senior", "5+ years", "6+ years", "7+ years", "advanced"],
            "lead": ["lead", "team lead", "technical lead", "principal", "architect"],
            "manager": ["manager", "management", "director", "head of"]
        }
        
        for level, patterns in experience_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return level
        
        return None
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better semantic matching.
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Lemmatize words
        words = word_tokenize(text)
        normalized_words = []
        
        for word in words:
            if word.lower() not in self.stop_words:
                lemmatized = self.lemmatizer.lemmatize(word.lower())
                normalized_words.append(lemmatized)
        
        return ' '.join(normalized_words)
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both texts
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        # Use spaCy for similarity calculation
        doc1 = self.nlp(norm_text1)
        doc2 = self.nlp(norm_text2)
        
        return doc1.similarity(doc2)

# Global text processor instance
text_processor = TextProcessor() 