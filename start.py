#!/usr/bin/env python3
"""
Startup script for the Semantic CV-Job Matching System
This script sets up the environment and starts the API server.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "transformers",
        "sentence-transformers", 
        "torch",
        "chromadb",
        "fastapi",
        "uvicorn",
        "spacy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def setup_environment():
    """Set up the environment and create necessary directories."""
    print("\n🔧 Setting up environment...")
    
    # Create necessary directories
    directories = [
        "vector_db",
        "uploads", 
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Check if spaCy model is installed
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model is installed")
    except OSError:
        print("⚠️  spaCy model not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                         check=True, capture_output=True)
            print("✅ spaCy model installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install spaCy model: {e}")
            return False
    
    return True

def run_tests():
    """Run system tests to ensure everything is working."""
    print("\n🧪 Running system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ System tests passed")
            return True
        else:
            print("❌ System tests failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ System tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def start_server():
    """Start the FastAPI server."""
    print("\n🚀 Starting Semantic CV-Job Matching System...")
    print("=" * 60)
    print("📡 API Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("📖 ReDoc Documentation: http://localhost:8000/redoc")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the server
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main startup function."""
    print("🎯 Semantic CV-Job Matching System - Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed.")
        return
    
    # Run tests (optional)
    print("\n💡 Would you like to run system tests? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            if not run_tests():
                print("\n⚠️  Tests failed, but you can still start the server.")
                print("Continue anyway? (y/n): ", end="")
                response = input().lower().strip()
                if response not in ['y', 'yes']:
                    return
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
        return
    
    # Start server
    start_server()

if __name__ == "__main__":
    main() 