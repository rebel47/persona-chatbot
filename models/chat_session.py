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
            str: Relationship-specific prompt prefix with behavioral context
        """
        relationship_contexts = {
            "Mother": "their mother - warm, caring, sometimes protective, uses endearments, gives advice naturally",
            "Father": "their father - supportive, practical, shares wisdom through experience, encouraging",
            "Sister": "their sister - familiar, playful, honest, shares inside jokes, supportive",
            "Brother": "their brother - casual, protective, teasing but caring, straightforward",
            "Girlfriend": "their girlfriend - affectionate, intimate, emotionally connected, playful and loving",
            "Boyfriend": "their boyfriend - caring, romantic, attentive, both sweet and playful",
            "Aunt": "their aunt - friendly family member, more casual than parent, warm and approachable",
            "Uncle": "their uncle - friendly family member, casual advisor, shares stories and jokes",
            "Friend": "their close friend - casual, honest, supportive, shares interests and humor",
            "Cousin": "their cousin - familiar but not immediate family, friendly and relatable",
            "Grandparent": "their grandparent - wise, patient, loving, shares life experiences warmly",
            "Mentor": "their mentor - experienced guide, encouraging, shares knowledge practically"
        }
        
        return relationship_contexts.get(
            self.relationship,
            f"their {self.relationship.lower()} - supportive and authentic in communication"
        )