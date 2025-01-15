import streamlit as st
import re
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

# Gemini model configuration
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

class ChatSession:
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.chat_history: List[Dict[str, str]] = []
        self.is_trained = False
        self.db = None
        self.model = None
        self.gemini_chat = None

    def add_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def initialize_gemini_chat(self):
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=GENERATION_CONFIG
        )
        self.gemini_chat = self.model.start_chat(history=[])

class RAGChatbot:
    def __init__(self):
        self.initialize_session_state()
        self.setup_ui()

    @staticmethod
    def initialize_session_state():
        if "chat_sessions" not in st.session_state:
            st.session_state["chat_sessions"] = {}
        if "current_session" not in st.session_state:
            st.session_state["current_session"] = None

    def setup_ui(self):
        st.title("Persona Chatbot: Emulate Chat Tone")
        st.text("       - Developed By: Mohammad Ayaz Alam")
        self.setup_sidebar()
        self.setup_chat_interface()

    def setup_sidebar(self):
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload WhatsApp Chat Log", type=["txt"])
            if uploaded_file:
                self.process_uploaded_file(uploaded_file)

            # Session selector
            self.setup_session_selector()

    def process_uploaded_file(self, uploaded_file):
        try:
            lines = uploaded_file.read().decode("utf-8").splitlines()
            messages = self.parse_messages(lines)
            senders = sorted(set(msg["sender"] for msg in messages))
            target_name = st.selectbox("Select the target person to emulate:", senders)

            if target_name and st.button("Start Training"):
                self.train_model(target_name, messages)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"File processing error: {str(e)}", exc_info=True)

    @staticmethod
    def parse_messages(lines: List[str]) -> List[Dict[str, str]]:
        message_pattern = re.compile(r"^(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (.*?): (.*)$")
        messages = []
        for line in lines:
            if match := message_pattern.match(line):
                messages.append({
                    "timestamp": match.groups()[0],
                    "sender": match.groups()[1],
                    "message": match.groups()[2]
                })
        return messages

    def train_model(self, target_name: str, messages: List[Dict[str, str]]):
        session_id = self.start_new_session(target_name)
        formatted_data = self.format_training_data(messages, target_name)
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
            documents = [
                Document(page_content=f"{item['prompt']} {item['completion']}")
                for item in formatted_data
            ]
            db = FAISS.from_documents(documents=documents, embedding=embeddings)
            
            session = st.session_state["chat_sessions"][session_id]
            session.db = db
            session.initialize_gemini_chat()
            session.is_trained = True
            st.success("Training completed! Start chatting below.")
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            logger.error(f"Training error: {str(e)}", exc_info=True)

    @staticmethod
    def format_training_data(messages: List[Dict[str, str]], target_name: str) -> List[Dict[str, str]]:
        formatted_data = []
        current_prompt = None
        previous_sender = None
        previous_message = None

        for msg in messages:
            if msg["sender"] == target_name:
                if previous_sender == target_name:
                    current_prompt += " " + msg["message"]
                else:
                    if current_prompt:
                        formatted_data.append({
                            "prompt": current_prompt,
                            "completion": previous_message["message"] if previous_message else ""
                        })
                    current_prompt = msg["message"]
            elif current_prompt:
                formatted_data.append({
                    "prompt": current_prompt,
                    "completion": msg["message"]
                })
                current_prompt = None
            previous_sender = msg["sender"]
            previous_message = msg

        return formatted_data

    def start_new_session(self, target_name: str) -> str:
        session_id = f"{target_name}_{len(st.session_state['chat_sessions']) + 1}"
        st.session_state["chat_sessions"][session_id] = ChatSession(target_name)
        st.session_state["current_session"] = session_id
        return session_id

    def setup_session_selector(self):
        session_ids = list(st.session_state["chat_sessions"].keys())
        if session_ids:
            selected_session = st.selectbox("Select a chat session:", session_ids)
            if selected_session:
                st.session_state["current_session"] = selected_session

    def setup_chat_interface(self):
        if st.session_state["current_session"]:
            session = st.session_state["chat_sessions"][st.session_state["current_session"]]
            st.subheader(f"Chat with {session.target_name}")

            chat_container = st.container()
            with chat_container:
                for chat in session.chat_history:
                    st.chat_message(chat["role"]).write(chat["content"])

            if prompt := st.chat_input("Type your message..."):
                self.process_user_message(prompt, session)

    def process_user_message(self, prompt: str, session: ChatSession):
        session.add_message("user", prompt)
        st.chat_message("user").write(prompt)

        with st.spinner("Generating response..."):
            try:
                bot_response = self.generate_response(prompt, session)
                session.add_message("assistant", bot_response)
                st.chat_message("assistant").write(bot_response)
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)

    @staticmethod
    def generate_response(question: str, session: ChatSession) -> str:
        docs = session.db.similarity_search(question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are {session.target_name}. Respond to the question below as {session.target_name} would, based on the provided chat history and context. Reflect {session.target_name}'s tone, personality, and style of communication. Adapt your responses to the mood and context—whether friendly, angry, emotional, or funny—while keeping them natural, engaging, lovable and concise. Do not mention or imply that you are anything other than {session.target_name}. If asked who or what you are, respond as {session.target_name} would.  If asked who or what you are, avoid any mention of being an AI, large language model, program, or anything similar.

Chat History: {json.dumps(session.chat_history)}
Context: {context}
Question: {question}
"""
        response = session.gemini_chat.send_message(prompt)
        return response.text

if __name__ == "__main__":
    chatbot = RAGChatbot()
