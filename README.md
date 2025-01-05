# Persona Chatbot: Emulate Conversational Style

## Overview

This project is an AI-powered chatbot that mimics the conversational style of a selected person based on their WhatsApp chat history. Users can upload a WhatsApp chat log, select a persona, and interact with a bot that generates responses emulating the tone, personality, and communication style of the chosen individual.

Try it live here: [Persona Chatbot](https://persona-chat.streamlit.app/)

## Features

- **WhatsApp Chat Preprocessing**: Parse and clean uploaded WhatsApp chat logs for easy processing.
- **Persona Selection**: Choose between two participants in the chat for persona emulation.
- **Conversational Style Emulation**: The bot uses the selected person's message history to emulate their style, tone, and manner of speaking.
- **Interactive Frontend**: User-friendly interface built with Streamlit, allowing users to upload chat logs, select a persona, and chat with the bot in real-time.
- **Contextual Chat**: Leverages the **Google Gemini API** for generating context-aware responses based on conversation history.

## Tech Stack

### Backend
- **Python**: Core programming language used for app development.
- **LangChain**: Facilitates the **Retrieval-Augmented Generation (RAG)** pipeline for efficient document retrieval and response generation.
- **FAISS/ChromaDB**: Vector databases for storing and searching message embeddings.
- **Gemini API**: Language model used for generating responses that emulate the selected persona’s conversational style.

### Frontend
- **Streamlit**: A simple, interactive UI framework for building web applications.

### Deployment
- **Hugging Face Spaces**: Free hosting for the application.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rebel47/persona-chatbot.git
   cd persona-chatbot
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app locally:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload a `.txt` file containing WhatsApp chat logs.
2. Select the persona (Person A or Person B) you want the chatbot to emulate from the dropdown.
3. Start interacting with the chatbot, which will generate responses that mimic the selected persona’s conversational style.

## File Structure

```
whatsapp-persona-chatbot/
├── app.py                # Main application script
├── preprocessing.py      # Preprocessing WhatsApp chat logs
├── fine_tuning.py        # Persona data preparation for training the chatbot
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

## Contributions

Feel free to contribute to this project by submitting a pull request or reporting issues. Please follow the guidelines in the `CONTRIBUTING.md` file.

## License

This project is licensed under the MIT License.
