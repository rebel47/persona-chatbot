# 💬 Persona Chatbot

AI chatbot that emulates conversational styles from WhatsApp chat history.

## Features

- 🎭 Emulates specific person's chat style
- 👨‍👩‍👧‍👦 Multiple relationship types (Mother, Father, Friend, etc.)
- 🧠 RAG-powered responses using chat context
- 💾 Multi-session support
- 🚀 Local embeddings (unlimited, free)

## Tech Stack

- **Streamlit** - Web interface
- **Google Gemini** - Text generation
- **HuggingFace** - Local embeddings
- **FAISS** - Vector search
- **LangChain** - RAG framework

## Installation

```bash
# Clone repository
git clone https://github.com/rebel47/persona-chatbot.git
cd persona-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run app
streamlit run app.py
```

## Usage

1. Upload WhatsApp chat export (.txt)
2. Select person to emulate
3. Choose relationship type
4. Click "Start Training"
5. Start chatting!

## Deployment

Deploy on Streamlit Cloud:
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Add `GOOGLE_API_KEY` in secrets
4. Deploy!

## License

MIT License

## Author

Mohammad Ayaz Alam ([rebel47](https://github.com/rebel47))

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
