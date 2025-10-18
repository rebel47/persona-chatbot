from typing import List, Dict, Optional
import streamlit as st
import json
import logging
import time
import google.api_core.exceptions
from google.genai import errors as genai_errors
from models.chat_session import ChatSession
from utils.message_parser import MessageParser
from utils.embeddings import EmbeddingManager
from config.settings import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot service."""
        self.embedding_manager = EmbeddingManager()
        self.initialize_session_state()
        self.message_parser = MessageParser(use_ai=False)  # ⚡ Fast regex parsing by default
        self.setup_ui()

    @staticmethod
    def initialize_session_state() -> None:
        """Initialize Streamlit session state variables."""
        if "chat_sessions" not in st.session_state:
            st.session_state["chat_sessions"] = {}
        if "current_session" not in st.session_state:
            st.session_state["current_session"] = None

    def setup_ui(self) -> None:
        """Set up the Streamlit user interface."""
        # Minimal custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #1E88E5;
        }
        .stButton button {
            width: 100%;
        }
        div[data-testid="stFileUploader"] label {
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="main-header">💬 Persona Chat</p>', unsafe_allow_html=True)
        
        self.setup_sidebar()
        self.setup_chat_interface()

    def setup_sidebar(self) -> None:
        """Set up the sidebar with file upload and session selection."""
        with st.sidebar:
            st.markdown("### Setup New Chat")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload WhatsApp Export",
                type=["txt"],
                help="Export chat from WhatsApp (without media)"
            )
            
            if uploaded_file:
                self.process_uploaded_file(uploaded_file)

            # Session management
            if st.session_state["chat_sessions"]:
                st.markdown("---")
                st.markdown("### Your Chats")
                self.setup_session_selector()

    def process_uploaded_file(self, uploaded_file) -> None:
        """Process the uploaded chat log file with relationship selection."""
        try:
            # ⚡ Show parsing progress
            with st.spinner("📖 Reading chat file..."):
                lines = uploaded_file.read().decode("utf-8").splitlines()
            
            with st.spinner("🔍 Parsing messages (smart detection)..."):
                messages = self.message_parser.parse_messages(lines, use_ai=False)  # Fast regex first
            
            if not messages:
                st.warning("⚠️ Standard format not detected. Trying AI-powered parsing...")
                with st.spinner("🤖 AI analyzing chat format... (10-20 seconds)"):
                    messages = self.message_parser.parse_messages(lines, use_ai=True)
            
            if not messages:
                st.error("❌ No valid messages found. Please check the file format.")
                return
            
            st.success(f"✅ Successfully parsed {len(messages)} messages!")
                
            senders = sorted(set(msg["sender"] for msg in messages))
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_name = st.selectbox(
                    "Select person",
                    senders,
                    help="Choose the person to emulate"
                )
            
            with col2:
                relationship_type = self.get_relationship_input()
            
            if target_name and relationship_type:
                if st.button("🚀 Start Training", use_container_width=True):
                    self.train_model(
                        target_name=target_name,
                        messages=messages,
                        relationship=relationship_type
                    )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"File processing error: {str(e)}", exc_info=True)

    @staticmethod
    def get_relationship_input() -> str:
        """Get the relationship type from user input."""
        common_relationships = [
            "Select relationship",
            "Mother",
            "Father",
            "Sister",
            "Brother",
            "Girlfriend",
            "Boyfriend",
            "Aunt",
            "Uncle",
            "Friend",
            "Cousin",
            "Grandparent",
            "Mentor",
            "Custom..."
        ]
        
        relationship = st.selectbox(
            "Choose relationship",
            common_relationships,
            help="Select the relationship type"
        )
        
        if relationship == "Custom...":
            custom_relationship = st.text_input(
                "Enter custom relationship",
                help="Type any relationship"
            )
            return custom_relationship if custom_relationship else ""
        
        return "" if relationship == "Select relationship" else relationship

    def train_model(self, target_name: str, messages: List[Dict[str, str]], relationship: str) -> None:
        """Train the model with the chat data."""
        try:
            session_id = self.start_new_session(target_name, relationship)
            formatted_data = self.format_training_data(messages, target_name)
            
            if not formatted_data:
                st.error(f"No training data found for {target_name}")
                return
            
            # ⚡ Show detailed progress with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"📊 Extracted {len(formatted_data)} training examples for {target_name}")
            progress_bar.progress(25)
            
            status_text.text(f"🧠 Creating embeddings... (this may take 20-30 seconds)")
            progress_bar.progress(50)
            
            vector_store = self.embedding_manager.create_vector_store(formatted_data)
            progress_bar.progress(75)
            
            status_text.text(f"✨ Initializing AI model...")
            session = st.session_state["chat_sessions"][session_id]
            session.db = vector_store
            session.initialize_gemini_chat()
            session.is_trained = True
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Ready! You can now chat with {relationship} {target_name}")
                
        except genai_errors.ClientError as e:
            # Handle Google GenAI SDK errors
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                st.error(f"💤 **{target_name} is taking a nap!**\n\nThe AI service needs a break. Come back later or get a new API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
            else:
                st.error(f"Something went wrong: {error_msg}")
            logger.error(f"Google GenAI error during training: {error_msg}", exc_info=True)
            
        except google.api_core.exceptions.ResourceExhausted as e:
            st.error(f"💤 {target_name} is offline right now. API quota exceeded - please try again later!")
            logger.error(f"API quota exceeded: {str(e)}", exc_info=True)
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            logger.error(f"Training error: {str(e)}", exc_info=True)

    def format_training_data(self, messages: List[Dict[str, str]], target_name: str) -> List[Dict[str, str]]:
        """Format chat messages into training data - only using target person's messages."""
        formatted_data = []
        
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]
            
            # Only include if the response is from the target person
            if next_msg["sender"] == target_name:
                formatted_data.append({
                    "prompt": current_msg["message"],
                    "completion": next_msg["message"]
                })
        
        logger.info(f"Extracted {len(formatted_data)} training examples for {target_name}")
        return formatted_data

    def start_new_session(self, target_name: str, relationship: str) -> str:
        """Start a new chat session."""
        session_id = f"{relationship}_{target_name}_{len(st.session_state['chat_sessions']) + 1}"
        st.session_state["chat_sessions"][session_id] = ChatSession(
            target_name=target_name,
            relationship=relationship
        )
        st.session_state["current_session"] = session_id
        return session_id

    def setup_session_selector(self) -> None:
        """Set up the session selector."""
        if st.session_state["chat_sessions"]:
            session_ids = list(st.session_state["chat_sessions"].keys())
            
            selected_session = st.selectbox(
                "Active Chat",
                session_ids,
                format_func=lambda x: f"{st.session_state['chat_sessions'][x].relationship} {st.session_state['chat_sessions'][x].target_name}",
                index=session_ids.index(st.session_state["current_session"]) if st.session_state["current_session"] in session_ids else 0,
                label_visibility="visible"
            )
            
            if selected_session:
                st.session_state["current_session"] = selected_session
                
                if st.button("🗑️ Clear History", use_container_width=True):
                    session = st.session_state["chat_sessions"][selected_session]
                    session.clear_history()
                    st.rerun()

    def setup_chat_interface(self) -> None:
        """Set up the main chat interface."""
        if st.session_state["current_session"]:
            session = st.session_state["chat_sessions"][st.session_state["current_session"]]
            
            if not session.is_trained:
                st.info("👋 Upload a WhatsApp chat to begin")
                return
            
            # Chat messages
            for msg in session.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Chat input
            if prompt := st.chat_input(f"Message {session.target_name}..."):
                self.process_user_message(prompt, session)

    def process_user_message(self, prompt: str, session: ChatSession) -> None:
        """Process a user message and generate a response."""
        try:
            # Add user message
            session.add_message("user", prompt)
            with st.chat_message("user"):
                st.write(prompt)

            # Show typing indicator
            with st.chat_message("assistant"):
                typing_placeholder = st.empty()
                typing_placeholder.markdown(f"*{session.target_name} is typing...*")
                
                try:
                    # ⚡ STREAMING RESPONSE for instant feedback
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Generate and stream response
                    for chunk in self.generate_response_stream(prompt, session):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")  # Blinking cursor
                    
                    typing_placeholder.empty()
                    response_placeholder.markdown(full_response)
                    session.add_message("assistant", full_response)
                    
                except google.api_core.exceptions.ResourceExhausted as e:
                    typing_placeholder.empty()
                    st.write(f"💤 {session.target_name} is offline. Try again later!")
                    logger.error(f"API quota exceeded: {str(e)}", exc_info=True)
                except Exception as e:
                    typing_placeholder.empty()
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error: {str(e)}", exc_info=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Error processing message: {str(e)}", exc_info=True)

    def generate_response(self, question: str, session: ChatSession) -> str:
        """Generate a response using the trained model with chain-of-thought."""
        try:
            docs = self.embedding_manager.get_similar_documents(question, session.db)
            
            # Extract examples for better context
            examples = []
            for doc in docs[:3]:  # Top 3 most relevant
                parts = doc.page_content.split(maxsplit=1)
                if len(parts) == 2:
                    examples.append(f"Q: {parts[0]}\nA: {parts[1]}")
            
            context_examples = "\n\n".join(examples)

            prompt = f"""You are {session.target_name}, {session.generate_prompt_prefix()}.

IMPORTANT: Respond EXACTLY as {session.target_name} would - matching their tone, style, vocabulary, and personality.

Here's how {session.target_name} typically responds:
{context_examples}

Recent conversation:
{json.dumps(session.chat_history[-3:], indent=2) if session.chat_history else "None"}

Current message: "{question}"

Think step by step:
1. What is the person asking/saying?
2. How would {session.target_name} feel about this?
3. What would {session.target_name} typically say in this situation?
4. What tone and words would {session.target_name} use?

Now respond as {session.target_name} (ONLY give the response, no thinking process):"""
            
            response = session.gemini_chat.send_message(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise
    
    def generate_response_stream(self, question: str, session: ChatSession):
        """
        ⚡ STREAMING version - yields response chunks as they're generated.
        Provides instant visual feedback to users.
        """
        try:
            docs = self.embedding_manager.get_similar_documents(question, session.db)
            
            # Extract examples for better context
            examples = []
            for doc in docs[:3]:  # Top 3 most relevant
                parts = doc.page_content.split(maxsplit=1)
                if len(parts) == 2:
                    examples.append(f"Q: {parts[0]}\nA: {parts[1]}")
            
            context_examples = "\n\n".join(examples)

            prompt = f"""You are {session.target_name}, {session.generate_prompt_prefix()}.

IMPORTANT: Respond EXACTLY as {session.target_name} would - matching their tone, style, vocabulary, and personality.

Here's how {session.target_name} typically responds:
{context_examples}

Recent conversation:
{json.dumps(session.chat_history[-3:], indent=2) if session.chat_history else "None"}

Current message: "{question}"

Think step by step:
1. What is the person asking/saying?
2. How would {session.target_name} feel about this?
3. What would {session.target_name} typically say in this situation?
4. What tone and words would {session.target_name} use?

Now respond as {session.target_name} (ONLY give the response, no thinking process):"""
            
            # Stream response chunks
            response = session.gemini_chat.send_message(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {str(e)}", exc_info=True)
            raise