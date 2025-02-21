import streamlit as st
import logging
from services.rag_chatbot import RAGChatbot
from config.settings import LOG_LEVEL, LOG_FORMAT
import google.api_core.exceptions

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)

logger = logging.getLogger(__name__)

def main():
    """
    Main application entry point.
    """
    try:
        # Set page configuration
        st.set_page_config(
            page_title="Persona Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize and run the chatbot
        chatbot = RAGChatbot()
        
    except google.api_core.exceptions.ResourceExhausted as e:
        logger.error(f"API quota exceeded: {str(e)}", exc_info=True)
        st.error("""
        ⚠️ API quota exceeded. This can happen when:
        - You've reached your daily API limit
        - Too many requests were made in a short time
        
        Solutions:
        1. Wait a few minutes and try again
        2. Use a different API key
        3. Contact support if the issue persists
        """)
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("""
        An error occurred while starting the application.
        Please check the logs or contact support.
        """)

if __name__ == "__main__":
    main()