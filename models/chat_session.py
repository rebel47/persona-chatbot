from typing import List, Dict, Optional
from dataclasses import dataclass, field
import google.generativeai as genai
import google.api_core.exceptions
from langchain_community.vectorstores import FAISS
from config.settings import GENERATION_CONFIG, CHAT_MODEL, GEMINI_API_KEY
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class ChatSession:
    """
    Represents a chat session with a specific target persona.
    """
    target_name: str
    relationship: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    is_trained: bool = False
    db: Optional[FAISS] = None
    model: Optional[genai.GenerativeModel] = None
    gemini_chat: Optional[any] = None
    last_request_time: float = field(default_factory=time.time)
    request_count: int = field(default=0)
    
    def initialize_gemini_chat(self) -> None:
        """
        Initialize the Gemini chat model with configuration.
        """
        try:
            self.model = genai.GenerativeModel(
                model_name=CHAT_MODEL,
                generation_config=GENERATION_CONFIG
            )
            self.gemini_chat = self.model.start_chat(history=[])
            logger.info(f"Successfully initialized Gemini chat for {self.target_name}")
        except google.api_core.exceptions.ResourceExhausted as e:
            logger.error(f"API quota exceeded during initialization: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini chat: {str(e)}")
            raise

    def check_rate_limit(self) -> None:
        """
        Check if we're within rate limits.
        Raises:
            ResourceExhausted: If rate limit is exceeded
        """
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        # Reset counter if more than 60 seconds have passed
        if time_diff > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Check if we've exceeded rate limit (e.g., 60 requests per minute)
        if self.request_count >= 60:
            raise google.api_core.exceptions.ResourceExhausted(
                "Rate limit exceeded. Please wait before sending more messages."
            )
            
        self.request_count += 1

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role (str): The role of the message sender (user/assistant)
            content (str): The content of the message
        """
        self.chat_history.append({
            "role": role,
            "content": content
        })


    def clear_history(self) -> None:
        """
        Clear the chat history while maintaining the session.
        """
        self.chat_history.clear()

    def generate_prompt_prefix(self) -> str:
        """
        Generate a context-aware prompt prefix based on the relationship.
        
        Returns:
            str: Relationship-specific prompt prefix
        """
        relationship_contexts = {
            "Mother": "a caring and nurturing mother who shows maternal love and guidance",
            "Father": "a supportive and guiding father who provides paternal wisdom and care",
            "Sister": "a close and understanding sister who shares family bonds and experiences",
            "Brother": "a protective and friendly brother who offers sibling support and companionship",
            "Girlfriend": "a loving and caring girlfriend in a romantic relationship",
            "Boyfriend": "an attentive and caring boyfriend in a romantic relationship",
            "Aunt": "a warm and caring aunt who is part of the extended family",
            "Uncle": "a friendly and wise uncle who shares family wisdom and experiences",
            "Friend": "a close and reliable friend who shares mutual trust and understanding",
            "Cousin": "a relatable cousin who shares family connections and experiences",
            "Grandparent": "a loving grandparent who shares wisdom and family history",
            "Mentor": "a guiding mentor who provides wisdom and professional advice"
        }
        
        return relationship_contexts.get(
            self.relationship,
            f"someone in the role of {self.relationship.lower()} who provides support and understanding"
        )