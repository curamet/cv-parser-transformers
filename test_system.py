#!/usr/bin/env python3
"""
Simple test script for the Semantic CV-Job Matching System
This script tests the core components without requiring the API server.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_text_processing():
    """Test text processing utilities."""
    print("🧪 Testing Text Processing...")
    
    try:
        from utils.text_processing import text_processor
        
        # Test text cleaning
        test_text = "  Hello   World!!!   "
        cleaned = text_processor.clean_text(test_text)
        print(f"Text cleaning: '{test_text}' -> '{cleaned}'")
        
        # Test section extraction
        cv_text = """
        SKILLS
        Python, JavaScript, React
        
        EXPERIENCE
        Senior Developer at TechCorp
        
        EDUCATION
        Computer Science Degree
        """
        sections = text_processor.extract_sections(cv_text)
        print(f"Sections extracted: {list(sections.keys())}")
        
        # Test skills extraction
        skills_text = "Python, JavaScript, React, AWS, Docker"
        skills = text_processor.extract_skills(skills_text)
        print(f"Skills extracted: {skills}")
        
        print("✅ Text processing tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Text processing tests failed: {e}")
        return False

def test_document_parser():
    """Test document parsing functionality."""
    print("\n🧪 Testing Document Parser...")
    
    try:
        from services.document_parser import document_parser
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test CV content\nSkills: Python, JavaScript\nExperience: 5 years")
            temp_file = f.name
        
        # Test document validation
        is_valid, message = document_parser.validate_document(temp_file)
        print(f"Document validation: {is_valid} - {message}")
        
        # Test document processing
        processed = document_parser._process_single_document(temp_file, "cv")
        if processed:
            print(f"Document processed successfully: {processed['doc_id']}")
        
        # Cleanup
        os.unlink(temp_file)
        
        print("✅ Document parser tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Document parser tests failed: {e}")
        return False

def test_nlp_processor():
    """Test NLP processing functionality."""
    print("\n🧪 Testing NLP Processor...")
    
    try:
        from services.nlp_processor import nlp_processor
        
        # Test embedding generation
        test_texts = ["Python developer", "Backend engineer"]
        embeddings = nlp_processor.generate_embeddings(test_texts)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Test similarity calculation
        similarity = nlp_processor.calculate_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between texts: {similarity:.3f}")
        
        # Test semantic features extraction
        features = nlp_processor.extract_semantic_features("Python developer with 5 years experience")
        print(f"Semantic features: {list(features.keys())}")
        
        print("✅ NLP processor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ NLP processor tests failed: {e}")
        return False

def test_vector_store():
    """Test vector storage functionality."""
    print("\n🧪 Testing Vector Store...")
    
    try:
        from services.vector_processor import vector_processor
        
        # Test collection stats
        stats = vector_processor.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Test health check
        health = vector_processor.get_collection_stats()
        print(f"Vector processor health: {health}")
        
        print("✅ Vector store tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Vector store tests failed: {e}")
        return False

def test_cv_processor():
    """Test CV processing pipeline."""
    print("\n🧪 Testing CV Processor...")
    
    try:
        from services.document_processor import document_processor
        
        # Create a temporary CV file
        cv_content = """
        SKILLS
        Python, JavaScript, React, AWS
        
        EXPERIENCE
        Senior Developer at TechCorp (2020-2023)
        - Led development team of 5 developers
        - Implemented microservices architecture
        
        EDUCATION
        Bachelor of Computer Science
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(cv_content)
            temp_file = f.name
        
        # Test CV processing
        processed_cvs = document_processor.process_documents_pipeline([temp_file], "cv")
        if processed_cvs:
            cv_data = processed_cvs[0]
            print(f"CV processed successfully: {cv_data['doc_id']}")
            print(f"Sections found: {list(cv_data['sections'].keys())}")
        
        # Cleanup
        os.unlink(temp_file)
        
        print("✅ CV processor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ CV processor tests failed: {e}")
        return False

def test_job_processor():
    """Test job processing pipeline."""
    print("\n🧪 Testing Job Processor...")
    
    try:
        from services.document_processor import document_processor
        
        # Create a temporary job file
        job_content = """
        SENIOR BACKEND DEVELOPER
        
        REQUIREMENTS
        - 5+ years experience in Python and JavaScript
        - Experience with Django, React frameworks
        - Knowledge of AWS, Docker, Kubernetes
        
        RESPONSIBILITIES
        - Design and implement scalable backend services
        - Lead technical architecture decisions
        - Mentor junior developers
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(job_content)
            temp_file = f.name
        
        # Test job processing
        processed_jobs = document_processor.process_documents_pipeline([temp_file], "job")
        if processed_jobs:
            job_data = processed_jobs[0]
            print(f"Job processed successfully: {job_data['doc_id']}")
        print(f"Sections found: {list(job_data['sections'].keys())}")
        
        # Cleanup
        os.unlink(temp_file)
        
        print("✅ Job processor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Job processor tests failed: {e}")
        return False

def test_semantic_matcher():
    """Test semantic matching functionality."""
    print("\n🧪 Testing Semantic Matcher...")
    
    try:
        from services.semantic_processor import semantic_processor
        
        # Test scoring weights
        print(f"Scoring weights: {semantic_processor.scoring_weights}")
        print(f"Matching thresholds: {semantic_processor.matching_thresholds}")
        
        # Test match quality determination
        quality = semantic_processor._determine_match_quality(0.85)
        print(f"Match quality for 0.85: {quality}")
        
        print("✅ Semantic matcher tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Semantic matcher tests failed: {e}")
        return False

def test_configuration():
    """Test configuration settings."""
    print("\n🧪 Testing Configuration...")
    
    try:
        from config.settings import settings
        
        # Test model configuration
        model_name = settings.get_model_name()
        print(f"Default model: {model_name}")
        
        # Test embedding dimension
        dimension = settings.get_embedding_dimension(model_name)
        print(f"Embedding dimension: {dimension}")
        
        # Test scoring weights
        print(f"Scoring weights: {settings.SCORING_WEIGHTS}")
        
        print("✅ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration tests failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Semantic CV-Job Matching System - Component Tests")
    print("=" * 60)
    
    tests = [
        test_configuration,
        test_text_processing,
        test_document_parser,
        test_nlp_processor,
        test_vector_store,
        test_cv_processor,
        test_job_processor,
        test_semantic_matcher
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Start the API server: python main.py")
        print("2. Run the example usage: python examples/example_usage.py")
        print("3. Visit the API docs: http://localhost:8000/docs")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 