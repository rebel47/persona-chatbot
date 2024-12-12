# Persona Chatbot

## Overview
This project is a chatbot that emulates the conversational style of a selected person based on their WhatsApp chat history. Users can upload a WhatsApp chat log between two people, select the persona they want to interact with, and chat with a bot that mimics the style of the chosen individual.

---

## Features
- **WhatsApp Chat Preprocessing**: Parse and clean uploaded WhatsApp chat logs.
- **Persona Selection**: Choose between two participants for chatbot emulation.
- **Conversational Style Emulation**: The bot uses the selected person's message style to generate responses.
- **Interactive Frontend**: User-friendly chat interface with upload and dropdown options.
---

## Tech Stack
### Backend
- **Python**
- **LangChain**: For Retrieval-Augmented Generation (RAG) pipeline.
- **FAISS/ChromaDB**: Vector database for message embeddings.
- **Gemini API**: Language model for response generation.

### Frontend
- **Streamlit**: Simple and interactive UI for user interaction.

### Deployment
- **Hugging Face Spaces**: Free hosting for the application.

---

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

---

## Usage
1. Upload a `.txt` file of WhatsApp chat logs.
2. Select the persona (Person A or Person B) from the dropdown.
3. Start chatting with the bot that mimics the chosen persona.

---

## File Structure
```
whatsapp-persona-chatbot/
├── app.py                # Main application script
├── preprocessing.py      # Preprocessing WhatsApp chat logs
├── fine_tuning.py        # Fine-tuning or persona data preparation
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

---


## Contributions
Feel free to contribute to this project by submitting a pull request or reporting issues. Follow the guidelines in `CONTRIBUTING.md`.

---

## License
This project is licensed under the MIT License.
