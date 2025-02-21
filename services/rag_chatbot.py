from typing import List, Dict, Optional
import streamlit as st
import json
import logging
import emoji
import google.api_core.exceptions
from models.chat_session import ChatSession
from utils.message_parser import MessageParser
from utils.embeddings import EmbeddingManager
from collections import Counter
import re
import nltk

logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot service."""
        self.embedding_manager = EmbeddingManager()
        self.initialize_session_state()
        self.message_parser = MessageParser()
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
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #1E88E5;
        }
        .sub-text {
            font-size: 1rem;
            color: #666;
            font-style: italic;
            margin-bottom: 0.5rem;
        }
        .developer-info {
            font-size: 0.9rem;
            color: #555;
            margin-bottom: 2rem;
            font-style: italic;
        }
        .stButton button {
            background-color: #1E88E5;
            color: white;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .user-message {
            background-color: #E3F2FD;
        }
        .bot-message {
            background-color: #F5F5F5;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Show title and developer info only on the main page
        if not st.session_state["current_session"]:
            st.markdown('<p class="main-title">Persona Chatbot</p>', unsafe_allow_html=True)
            st.markdown('<p class="sub-text">Emulate chat style based on relationships</p>', unsafe_allow_html=True)
            st.markdown('<p class="developer-info">Developed by: Mohammad Ayaz Alam</p>', unsafe_allow_html=True)
            
            # Add a divider
            st.markdown("---")
        
        self.setup_sidebar()
        self.setup_chat_interface()

    def setup_sidebar(self) -> None:
        """Set up the sidebar with file upload and session selection."""
        with st.sidebar:
            st.markdown("### Create New Chat")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload WhatsApp Chat",
                type=["txt"],
                help="Select a WhatsApp chat export file (.txt format)"
            )
            
            if uploaded_file:
                self.process_uploaded_file(uploaded_file)

            st.markdown("---")
            
            # Session management
            self.setup_session_selector()

    def process_uploaded_file(self, uploaded_file) -> None:
        """Process the uploaded chat log file with relationship selection."""
        try:
            lines = uploaded_file.read().decode("utf-8").splitlines()
            messages = self.message_parser.parse_messages(lines)
            
            if not messages:
                st.error("No valid messages found in the file.")
                return
                
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
                if st.button("Start Training", help="Begin training the model", use_container_width=True):
                    with st.spinner(f"Training model to emulate {relationship_type} {target_name}..."):
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

    def analyze_speaking_style(self, messages: List[Dict[str, str]], target_name: str) -> Dict[str, any]:
        """Analyze the speaking style of the target person."""
        target_messages = [msg["message"] for msg in messages if msg["sender"] == target_name]
        
        if not target_messages:
            return {}
            
        # Analyze text
        all_text = " ".join(target_messages).lower()
        words = re.findall(r'\b\w+\b', all_text)
        
        # Common words (excluding stop words)
        common_words = Counter(words).most_common(20)
        
        # Analyze message characteristics
        style_characteristics = {
            "common_words": dict(common_words),
            "avg_message_length": sum(len(msg.split()) for msg in target_messages) / len(target_messages),
            "uses_emoji": any(emoji.emoji_count(msg) > 0 for msg in target_messages),
            "capitalization": sum(1 for msg in target_messages if msg.isupper()) / len(target_messages),
            "punctuation_style": {
                "exclamation": sum("!" in msg for msg in target_messages) / len(target_messages),
                "question": sum("?" in msg for msg in target_messages) / len(target_messages),
                "ellipsis": sum("..." in msg for msg in target_messages) / len(target_messages),
            },
            "common_phrases": self.extract_common_phrases(target_messages),
            "spelling_patterns": self.analyze_spelling_patterns(target_messages)
        }
        
        return style_characteristics

    def extract_common_phrases(self, messages: List[str], min_length: int = 2, max_length: int = 4) -> Dict[str, int]:
        """Extract commonly used phrases."""
        phrases = []
        for message in messages:
            words = message.lower().split()
            for i in range(len(words)):
                for j in range(min_length, min(max_length + 1, len(words) - i + 1)):
                    phrase = " ".join(words[i:i+j])
                    phrases.append(phrase)
        
        return dict(Counter(phrases).most_common(10))

    def analyze_spelling_patterns(self, messages: List[str]) -> Dict[str, str]:
        """Analyze common spelling patterns or intentional misspellings."""
        patterns = {}
        
        # Common text patterns to look for
        pattern_rules = {
            r'([haha]+)': 'laugh_pattern',
            r'([!?]+)': 'punctuation_pattern',
            r'(\w)\1{2,}': 'letter_repetition',  # e.g., "yeahhh"
            r'k\b': 'k_instead_of_ok',
            r'u\b': 'u_instead_of_you',
            r'r\b': 'r_instead_of_are',
        }
        
        for message in messages:
            for pattern, name in pattern_rules.items():
                if matches := re.findall(pattern, message.lower()):
                    patterns[name] = patterns.get(name, 0) + len(matches)
        
        return patterns

    def train_model(self, target_name: str, messages: List[Dict[str, str]], relationship: str) -> None:
        """Train the model with the chat data and style analysis."""
        try:
            # Analyze speaking style
            style_characteristics = self.analyze_speaking_style(messages, target_name)
            
            # Create new session
            session_id = self.start_new_session(target_name, relationship)
            session = st.session_state["chat_sessions"][session_id]
            session.style_characteristics = style_characteristics
            
            # Format training data
            formatted_data = self.format_training_data(messages, target_name)
            
            if not formatted_data:
                st.error(f"No training data could be extracted for {target_name}")
                return
            
            vector_store = self.embedding_manager.create_vector_store(formatted_data)
            
            session.db = vector_store
            session.initialize_gemini_chat()
            session.is_trained = True
            
            # Show style analysis
            with st.expander("Speaking Style Analysis"):
                st.write("Common Words:", style_characteristics["common_words"])
                st.write("Average Message Length:", f"{style_characteristics['avg_message_length']:.1f} words")
                st.write("Uses Emoji:", "Yes" if style_characteristics["uses_emoji"] else "No")
                st.write("Common Phrases:", style_characteristics["common_phrases"])
            
            st.success(f"✨ Training completed! You can now chat with your {relationship.lower()} {target_name}")
                
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            logger.error(f"Training error: {str(e)}", exc_info=True)

    def format_training_data(self, messages: List[Dict[str, str]], target_name: str) -> List[Dict[str, str]]:
        """Format chat messages into training data."""
        formatted_data = []
        conversation_window = []
        
        for msg in messages:
            conversation_window.append(msg)
            if len(conversation_window) >= 3:
                if conversation_window[1]["sender"] == target_name:
                    formatted_data.append({
                        "prompt": conversation_window[0]["message"],
                        "completion": f"{conversation_window[1]['message']} {conversation_window[2]['message']}"
                    })
                conversation_window.pop(0)
                
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
            st.markdown("### Your Chats")
            session_ids = list(st.session_state["chat_sessions"].keys())
            
            selected_session = st.selectbox(
                "Select a chat",
                session_ids,
                format_func=lambda x: f"{st.session_state['chat_sessions'][x].relationship} {st.session_state['chat_sessions'][x].target_name}",
                index=session_ids.index(st.session_state["current_session"]) if st.session_state["current_session"] in session_ids else 0
            )
            
            if selected_session:
                st.session_state["current_session"] = selected_session
                session = st.session_state["chat_sessions"][selected_session]
                
                if st.button("Clear Chat History", use_container_width=True):
                    session.clear_history()
                    st.rerun()  # Using st.rerun() instead of experimental_rerun

    def setup_chat_interface(self) -> None:
        """Set up the main chat interface."""
        if st.session_state["current_session"]:
            session = st.session_state["chat_sessions"][st.session_state["current_session"]]
            
            if not session.is_trained:
                st.info("👋 Please upload a chat log and train the model to start chatting.")
                return
                
            st.markdown(f"### Chat with {session.relationship} {session.target_name}")
            
            # Chat history
            for msg in session.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Chat input
            if prompt := st.chat_input(f"Message {session.relationship} {session.target_name}..."):
                self.process_user_message(prompt, session)

    def process_user_message(self, prompt: str, session: ChatSession) -> None:
        """Process a user message and generate a response."""
        try:
            # Add user message
            session.add_message("user", prompt)
            st.chat_message("user").write(prompt)

            with st.spinner("Generating response..."):
                try:
                    bot_response = self.generate_response(prompt, session)
                    session.add_message("assistant", bot_response)
                    st.chat_message("assistant").write(bot_response)
                except google.api_core.exceptions.ResourceExhausted as e:
                    error_message = ("⚠️ API quota exceeded. Please try again later or use a different API key. "
                                   "This usually happens when you've reached your daily limit.")
                    st.error(error_message)
                    logger.error(f"API quota exceeded: {str(e)}", exc_info=True)
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    logger.error(error_message, exc_info=True)
        except Exception as e:
            error_message = f"Error processing message: {str(e)}"
            st.error(error_message)
            logger.error(error_message, exc_info=True)

    def generate_response(self, question: str, session: ChatSession) -> str:
        """Generate a response using the trained model."""
        try:
            docs = self.embedding_manager.get_similar_documents(question, session.db)
            context = "\n".join([doc.page_content for doc in docs])
            
            style = session.style_characteristics
            style_instructions = f"""
            Match these speaking characteristics:
            - Use common words like: {', '.join(list(style['common_words'].keys())[:5])}
            - Typical message length: {style['avg_message_length']:.1f} words
            - {'Use emojis occasionally' if style['uses_emoji'] else 'Avoid emojis'}
            - Common phrases: {', '.join(list(style['common_phrases'].keys())[:3])}
            - Match spelling patterns: {style['spelling_patterns']}
            - Use punctuation style: {style['punctuation_style']}
            - {'Use ALL CAPS sometimes' if style['capitalization'] > 0.1 else 'Avoid ALL CAPS'}
            
            Important:
            1. Maintain the exact speaking style, including any misspellings or informal language
            2. Use the common phrases naturally in the conversation
            3. Match the message length pattern
            4. Keep the personality consistent with previous responses
            """

            prompt = f"""
            You are {session.target_name}, {session.generate_prompt_prefix()}. 
            {style_instructions}
            
            Respond naturally while maintaining these speaking patterns.
            Recent Chat History: {json.dumps(session.chat_history[-5:])}
            Similar Conversation Context: {context}
            Current Message: {question}
            
            Remember:
            - Never mention being an AI or chat model
            - Stay in character as {session.target_name}
            - Use the exact speaking style shown in the examples
            """
            
            response = session.gemini_chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise