import streamlit as st
import re
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title("RAG Chatbot: Emulate Chat Tone")

# Initialize session state variables
if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = {}

if "current_session" not in st.session_state:
    st.session_state["current_session"] = None

def start_new_session(target_name):
    session_id = f"{target_name}_{len(st.session_state['chat_sessions']) + 1}"
    st.session_state["chat_sessions"][session_id] = {
        "target_name": target_name,
        "chat_history": [],
        "is_trained": False,
        "db": None,
        "model": None
    }
    st.session_state["current_session"] = session_id
    return session_id

# Sidebar for configuration
with st.sidebar:
    uploaded_file = st.file_uploader("Upload WhatsApp Chat Log", type=["txt"])

    if uploaded_file:
        lines = uploaded_file.read().decode("utf-8").splitlines()
        message_pattern = re.compile(r"^(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (.*?): (.*)$")
        messages = [
            {"timestamp": match.groups()[0], "sender": match.groups()[1], "message": match.groups()[2]}
            for line in lines if (match := message_pattern.match(line))
        ]
        senders = sorted(set(msg["sender"] for msg in messages))
        target_name = st.selectbox("Select the target person to emulate:", senders)

        if target_name and st.button("Start Training"):
            session_id = start_new_session(target_name)
            formatted_data = []
            current_prompt = None
            previous_sender = None

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

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
            documents = [Document(page_content=f"{item['prompt']} {item['completion']}") for item in formatted_data]
            db = FAISS.from_documents(documents=documents, embedding=embeddings)
            model = genai.GenerativeModel('gemini-pro')

            st.session_state["chat_sessions"][session_id]["db"] = db
            st.session_state["chat_sessions"][session_id]["model"] = model
            st.session_state["chat_sessions"][session_id]["is_trained"] = True
            st.success("Training completed! Start chatting below.")

    # Session selector
    session_ids = list(st.session_state["chat_sessions"].keys())
    if session_ids:
        selected_session = st.selectbox("Select a chat session:", session_ids)
        if selected_session:
            st.session_state["current_session"] = selected_session

# Main chat interface
if st.session_state["current_session"]:
    session = st.session_state["chat_sessions"][st.session_state["current_session"]]

    def answer_question(question):
        db = session["db"]
        model = session["model"]
        target_name = session["target_name"]

        docs = db.similarity_search(question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are {target_name}. Respond to the question below as {target_name} would, based on the provided chat history and context. Reflect {target_name}'s tone, personality, and style of communication. Adapt your responses to the mood and context—whether friendly, angry, emotional, or funny—while keeping them natural, engaging, lovable and concise. Do not mention or imply that you are anything other than {target_name}. If asked who or what you are, respond as {target_name} would.  If asked who or what you are, avoid any mention of being an AI, large language model, program, or anything similar.
Chat History: {json.dumps(session['chat_history'])}
Context: {context}
Question: {question}
"""
        print(prompt)
        response = model.generate_content(prompt)
        return response.text

    st.subheader(f"Chat with {session['target_name']}")

    chat_container = st.container()
    with chat_container:
        for chat in session["chat_history"]:
            st.chat_message(chat["role"]).write(chat["content"])

    if prompt := st.chat_input("Type your message..."):
        session["chat_history"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Generating response..."):
            bot_response = answer_question(prompt)
        session["chat_history"].append({"role": "assistant", "content": bot_response})
        st.chat_message("assistant").write(bot_response)
