from config.settings import settings
import requests
import json
from typing import Optional, Dict, Any
from loguru import logger

async def call_llm(prompt: str, model: str = None, temperature: float = 0.7, max_tokens: int = 2048) -> Optional[str]:
    """
    Generic LLM API caller that works with multiple providers

    Args:
        prompt: The prompt to send to the model
        model: The model to use (defaults to LLM_MODEL from config)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens to generate

    Returns:
        The response from the model, or None if error
    """
    if model is None:
        model = settings.LLM_MODEL

    try:
        if settings.LLM_PROVIDER == "ollama":
            return await call_ollama_api(prompt, model, temperature, max_tokens)
        elif settings.LLM_PROVIDER == "openai":
            return await call_openai_api(prompt, model, temperature, max_tokens)
        else:
            logger.error(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
            return None

    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        return None


async def call_ollama_api(prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[str]:
    """Call Ollama API"""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": max_tokens
            }
        }

        response = requests.post(
            f"{settings.LLM_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error calling Ollama API: {str(e)}")
        return None


async def call_openai_api(prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[str]:
    """Call OpenAI API"""
    try:
        headers = {
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(
            f"{settings.LLM_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return None


async def analyze_cv_with_llm(cv_text: str) -> Optional[str]:
    """
    Analyze a CV using the LLM with the CV analysis prompt

    Args:
        cv_text: The raw CV text to analyze

    Returns:
        The analysis response from the LLM, or None if error
    """
    try:
        # Get the prompt and combine with CV text
        prompt = settings.CV_ANALYSIS_PROMPT + "\n\n" + cv_text

        # Call LLM with appropriate parameters for CV analysis
        response = await call_llm(
            prompt=prompt,
            temperature=0.7,  # Balanced creativity and consistency
            max_tokens=2048  # Enough for detailed analysis
        )

        return response

    except Exception as e:
        logger.error(f"Error analyzing CV with LLM: {str(e)}")
        return None


def parse_cv_analysis(analysis_text: str) -> Dict[str, Any]:
    """
    Parse the LLM analysis response into structured data

    Args:
        analysis_text: Raw text response from LLM

    Returns:
        Structured analysis data
    """
    try:
        # Initialize result structure
        result = {
            "categories": {},
            "interview_probability": "",
            "suggested_improvements": [],
            "summary": "",
            "raw_analysis": analysis_text
        }

        lines = analysis_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            # Parse category scores
            if line.startswith("**Technical Skills Match:"):
                score = line.split(":")[1].strip().replace("**", "")
                result["categories"]["technical_skills"] = {"score": score, "comments": ""}
                current_section = "technical_skills"
            elif line.startswith("**Experience Relevance:"):
                score = line.split(":")[1].strip().replace("**", "")
                result["categories"]["experience_relevance"] = {"score": score, "comments": ""}
                current_section = "experience_relevance"
            elif line.startswith("**Certifications:"):
                score = line.split(":")[1].strip().replace("**", "")
                result["categories"]["certifications"] = {"score": score, "comments": ""}
                current_section = "certifications"
            elif line.startswith("**Soft Skills & Language:"):
                score = line.split(":")[1].strip().replace("**", "")
                result["categories"]["soft_skills"] = {"score": score, "comments": ""}
                current_section = "soft_skills"
            elif line.startswith("**Quantifiable Achievements:"):
                score = line.split(":")[1].strip().replace("**", "")
                result["categories"]["quantifiable_achievements"] = {"score": score, "comments": ""}
                current_section = "quantifiable_achievements"
            elif line.startswith("**Tools & Ecosystem Use:"):
                score = line.split(":")[1].strip().replace("**", "")
                result["categories"]["tools_ecosystem"] = {"score": score, "comments": ""}
                current_section = "tools_ecosystem"

            # Parse interview probability
            elif line.startswith("~") and "%" in line:
                result["interview_probability"] = line
                current_section = None

            # Parse improvements
            elif line.startswith("**🔧 Suggested Improvements**"):
                current_section = "improvements"
            elif current_section == "improvements" and line.startswith(("1.", "2.", "3.", "4.", "5.", "6.")):
                result["suggested_improvements"].append(line)

            # Parse summary
            elif line.startswith("**✅ Summary**"):
                current_section = "summary"
            elif current_section == "summary" and line and not line.startswith("**"):
                result["summary"] += line + " "

            # Add comments to current category
            elif current_section in ["technical_skills", "experience_relevance", "certifications",
                                     "soft_skills", "quantifiable_achievements", "tools_ecosystem"]:
                if line and not line.startswith("**"):
                    result["categories"][current_section]["comments"] += line + " "

        # Clean up comments
        for category in result["categories"]:
            result["categories"][category]["comments"] = result["categories"][category]["comments"].strip()

        result["summary"] = result["summary"].strip()

        return result

    except Exception as e:
        logger.error(f"Error parsing CV analysis: {str(e)}")
        return {"error": "Failed to parse analysis", "raw_analysis": analysis_text}
