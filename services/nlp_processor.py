import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from loguru import logger
from config.settings import settings
from utils.text_processing import text_processor

class NLPProcessor:
    """NLP processing service using HuggingFace transformers for semantic understanding."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize NLP processor with specified model.
        
        Args:
            model_name: HuggingFace model name to use
        """
        self.model_name = model_name or settings.get_model_name()
        self.model = None
        self.tokenizer = None
        self.embedding_dimension = settings.get_embedding_dimension(self.model_name)
        
        # Initialize model
        self._load_model()
        
        # Cache for embeddings
        self.embedding_cache = {}
    
    def _load_model(self):
        """Load the specified HuggingFace model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            if "sentence-transformers" in self.model_name:
                # Use SentenceTransformers for better performance
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            else:
                # Use HuggingFace transformers
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                logger.info(f"Loaded HuggingFace model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            # Fallback to primary model
            fallback_model = settings.EMBEDDING_MODELS["primary"]
            logger.info(f"Falling back to: {fallback_model}")
            self.model_name = fallback_model
            self.model = SentenceTransformer(fallback_model)
    
    def generate_embeddings(self, texts: Union[str, List[str]], 
                          batch_size: int = None) -> np.ndarray:
        """
        Generate embeddings for given texts.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        batch_size = batch_size or settings.PROCESSING_SETTINGS["batch_size"]
        
        try:
            if isinstance(self.model, SentenceTransformer):
                # Use SentenceTransformers
                embeddings = self.model.encode(
                    texts, 
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            else:
                # Use HuggingFace transformers
                embeddings = self._generate_embeddings_hf(texts, batch_size)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.embedding_dimension))
    
    def _generate_embeddings_hf(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        Generate embeddings using HuggingFace transformers.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=settings.PROCESSING_SETTINGS["max_text_length"],
                return_tensors="pt"
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                attention_mask = inputs['attention_mask']
                embeddings_batch = self._mean_pooling(outputs.last_hidden_state, attention_mask)
                embeddings.append(embeddings_batch.numpy())
        
        return np.vstack(embeddings)
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, 
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings.
        
        Args:
            token_embeddings: Token embeddings from model
            attention_mask: Attention mask
            
        Returns:
            Mean pooled embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def calculate_similarity(self, embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def calculate_batch_similarity(self, query_embedding: np.ndarray, 
                                 candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between query and multiple candidates.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: Array of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        try:
            # Normalize embeddings
            query_norm = np.linalg.norm(query_embedding)
            candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
            
            # Avoid division by zero
            query_norm = max(query_norm, 1e-8)
            candidate_norms = np.maximum(candidate_norms, 1e-8)
            
            # Calculate cosine similarities
            similarities = np.dot(candidate_embeddings, query_embedding) / (candidate_norms * query_norm)
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating batch similarity: {str(e)}")
            return np.zeros(len(candidate_embeddings))
    
    def extract_semantic_features(self, text: str) -> Dict:
        """
        Extract semantic features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing semantic features
        """
        features = {
            "skills": [],
            "experience_level": None,
            "technical_terms": [],
            "entities": [],
            "sections": {}
        }
        
        try:
            # Extract sections
            sections = text_processor.extract_sections(text)
            features["sections"] = sections
            
            # Extract skills
            if sections.get("skills"):
                features["skills"] = text_processor.extract_skills(sections["skills"])
            
            # Extract experience level
            if sections.get("experience"):
                features["experience_level"] = text_processor.extract_experience_level(sections["experience"])
            
            # Extract technical terms using spaCy
            doc = text_processor.nlp(text)
            technical_terms = []
            entities = []
            
            for token in doc:
                # Technical terms (nouns, proper nouns)
                if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                    technical_terms.append(token.text)
                
                # Named entities
                if token.ent_type_:
                    entities.append({
                        "text": token.text,
                        "type": token.ent_type_,
                        "label": token.ent_type_
                    })
            
            features["technical_terms"] = list(set(technical_terms))
            features["entities"] = entities
            
        except Exception as e:
            logger.error(f"Error extracting semantic features: {str(e)}")
        
        return features
    
    def understand_semantic_relationships(self, text1: str, text2: str) -> Dict:
        """
        Understand semantic relationships between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary containing relationship analysis
        """
        relationships = {
            "overall_similarity": 0.0,
            "section_similarities": {},
            "skill_overlap": [],
            "experience_compatibility": False,
            "semantic_equivalence": []
        }
        
        try:
            # Calculate overall similarity
            embedding1 = self.generate_embeddings(text1)
            embedding2 = self.generate_embeddings(text2)
            relationships["overall_similarity"] = self.calculate_similarity(embedding1[0], embedding2[0])
            
            # Extract sections and compare
            sections1 = text_processor.extract_sections(text1)
            sections2 = text_processor.extract_sections(text2)
            
            for section_name in ["skills", "experience", "education", "projects"]:
                if sections1.get(section_name) and sections2.get(section_name):
                    section_emb1 = self.generate_embeddings(sections1[section_name])
                    section_emb2 = self.generate_embeddings(sections2[section_name])
                    similarity = self.calculate_similarity(section_emb1[0], section_emb2[0])
                    relationships["section_similarities"][f"{section_name}_match"] = similarity
            
            # Extract and compare skills
            skills1 = text_processor.extract_skills(sections1.get("skills", ""))
            skills2 = text_processor.extract_skills(sections2.get("skills", ""))
            
            # Find overlapping skills
            skills1_lower = [skill.lower() for skill in skills1]
            skills2_lower = [skill.lower() for skill in skills2]
            overlap = [skill for skill in skills1 if skill.lower() in skills2_lower]
            relationships["skill_overlap"] = overlap
            
            # Check experience compatibility
            exp_level1 = text_processor.extract_experience_level(sections1.get("experience", ""))
            exp_level2 = text_processor.extract_experience_level(sections2.get("experience", ""))
            
            if exp_level1 and exp_level2:
                # Simple compatibility check
                level_hierarchy = ["junior", "mid", "senior", "lead", "manager"]
                try:
                    idx1 = level_hierarchy.index(exp_level1)
                    idx2 = level_hierarchy.index(exp_level2)
                    relationships["experience_compatibility"] = abs(idx1 - idx2) <= 1
                except ValueError:
                    relationships["experience_compatibility"] = True
            
            # Find semantic equivalents
            semantic_equivalents = self._find_semantic_equivalents(text1, text2)
            relationships["semantic_equivalence"] = semantic_equivalents
            
        except Exception as e:
            logger.error(f"Error understanding semantic relationships: {str(e)}")
        
        return relationships
    
    def _find_semantic_equivalents(self, text1: str, text2: str) -> List[Tuple[str, str]]:
        """
        Find semantically equivalent terms between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            List of equivalent term pairs
        """
        equivalents = []
        
        try:
            # Extract technical terms
            doc1 = text_processor.nlp(text1)
            doc2 = text_processor.nlp(text2)
            
            terms1 = [token.text for token in doc1 if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2]
            terms2 = [token.text for token in doc2 if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2]
            
            # Find similar terms using embeddings
            if terms1 and terms2:
                embeddings1 = self.generate_embeddings(terms1)
                embeddings2 = self.generate_embeddings(terms2)
                
                # Calculate similarities
                similarities = self.calculate_batch_similarity(embeddings1[0], embeddings2)
                
                # Find high similarity pairs
                for i, term1 in enumerate(terms1):
                    for j, term2 in enumerate(terms2):
                        if similarities[j] > 0.8:  # High similarity threshold
                            equivalents.append((term1, term2))
        
        except Exception as e:
            logger.error(f"Error finding semantic equivalents: {str(e)}")
        
        return equivalents
    
    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding if available.
        
        Args:
            text: Input text
            
        Returns:
            Cached embedding or None
        """
        text_hash = hash(text)
        return self.embedding_cache.get(text_hash)
    
    def cache_embedding(self, text: str, embedding: np.ndarray):
        """
        Cache embedding for future use.
        
        Args:
            text: Input text
            embedding: Generated embedding
        """
        text_hash = hash(text)
        self.embedding_cache[text_hash] = embedding
        
        # Limit cache size
        if len(self.embedding_cache) > 10000:
            # Remove oldest entries
            keys_to_remove = list(self.embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                del self.embedding_cache[key]

# Global NLP processor instance
nlp_processor = NLPProcessor() 