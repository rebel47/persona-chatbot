import os
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Model Configuration
GENERATION_CONFIG: Dict[str, Any] = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

# Model Names
EMBEDDING_MODEL = "models/embedding-005"
CHAT_MODEL = "gemini-2.0-flash" 

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
