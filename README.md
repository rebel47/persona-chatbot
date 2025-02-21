# Persona Chatbot: Emulate Chat Tone

## Overview

A sophisticated AI chatbot that can emulate the conversational style of specific individuals based on their WhatsApp chat history. The bot can take on different relationship roles (mother, father, friend, etc.) while maintaining the authentic communication style of the selected person.

Last Updated: 2025-02-21 05:12:22 UTC

## Key Features

### Core Functionality
- **Relationship-Based Interactions**: Choose from various relationship types (parent, sibling, friend, etc.)
- **Style Emulation**: Accurately mimics the selected person's communication patterns
- **Context-Aware Responses**: Maintains conversation coherence using chat history
- **Multi-Session Support**: Create and manage multiple chat sessions

### Technical Features
- **RAG Implementation**: Uses Retrieval-Augmented Generation for accurate response generation
- **Vector Search**: FAISS-powered similarity search for relevant context retrieval
- **Robust Error Handling**: Graceful handling of API limits and errors
- **Clean, Modern UI**: Streamlit-based interface with custom styling

## Tech Stack

### Core Technologies
- **Python**: Primary development language
- **Google Gemini API**: Large language model for response generation
- **LangChain**: Framework for RAG implementation
- **FAISS**: Vector storage and similarity search
- **Streamlit**: Web interface framework

### Additional Libraries
- **google-generativeai**: Gemini API integration
- **langchain-google-genai**: LangChain integration for Gemini
- **python-dotenv**: Environment variable management
- **logging**: Comprehensive error tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rebel47/persona-chatbot.git
cd persona-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
```

5. Run the application:
```bash
streamlit run main.py
```

## Project Structure

```
persona-chatbot/
├── config/
│   └── settings.py           # Configuration and environment variables
├── models/
│   └── chat_session.py       # Chat session management
├── services/
│   └── rag_chatbot.py        # Core chatbot implementation
├── utils/
│   ├── embeddings.py         # Vector embedding utilities
│   └── message_parser.py     # WhatsApp message parsing
├── main.py                   # Application entry point
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Usage Guide

1. **Start the Application**
   - Run the application using `streamlit run main.py`
   - Access the web interface at `http://localhost:8501`

2. **Upload Chat Data**
   - Click "Upload WhatsApp Chat" in the sidebar
   - Select a WhatsApp chat export file (`.txt` format)

3. **Configure Chat**
   - Select the person to emulate from the chat
   - Choose or enter a relationship type
   - Click "Start Training" to initialize the chat

4. **Start Chatting**
   - Type messages in the chat input
   - Receive responses that match the selected person's style
   - Switch between different chat sessions as needed

## Error Handling

The application includes robust error handling for common issues:
- API quota management
- Rate limiting
- File processing errors
- Model initialization issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Developer

Developed by: Mohammad Ayaz Alam (rebel47)  
Contact: [GitHub Profile](https://github.com/rebel47)

## Acknowledgments

- Google Gemini API for providing the language model
- Streamlit for the excellent web framework
- The open-source community for various tools and libraries