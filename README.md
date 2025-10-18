# 💬 Persona Chatbot

A sophisticated AI chatbot that learns and emulates conversational styles from WhatsApp chat history using advanced RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

- 🎭 **Persona Emulation** - Mimics specific person's chat style, vocabulary, and tone
- 👨‍👩‍👧‍👦 **Relationship Context** - Multiple relationship types (Mother, Father, Friend, Partner, etc.)
- 🧠 **RAG-Powered** - Uses chain-of-thought reasoning with relevant chat examples
- 💾 **Multi-Session** - Manage multiple conversations simultaneously
- 🚀 **Local Embeddings** - HuggingFace transformers (unlimited, no API limits)
- 💬 **Typing Indicators** - Real-time chat experience
- 🎯 **Smart Caching** - Vector stores cached for instant reloading
- 📊 **Vector Explorer** - Jupyter notebook to explore embeddings

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Web Framework** | Streamlit 1.42.2 |
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | HuggingFace (sentence-transformers/all-MiniLM-L6-v2) |
| **Vector DB** | FAISS 1.12.0 with AVX2 |
| **RAG Framework** | LangChain 0.3.19 |
| **ML Backend** | PyTorch 2.9.0 |
| **Language** | Python 3.11+ |

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Google API Key ([Get one here](https://makersuite.google.com/app/apikey))
- WhatsApp chat export file

### Installation

```bash
# Clone repository
git clone https://github.com/rebel47/persona-chatbot.git
cd persona-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📖 How to Use

### 1. Export WhatsApp Chat
- Open WhatsApp chat
- Tap ⋮ (menu) → More → Export chat
- Choose "Without Media"
- Save the `.txt` file

### 2. Train Your Persona
1. Upload the exported chat file
2. Select the person you want to emulate from dropdown
3. Choose relationship type (affects conversation context)
4. Click **"Start Training"**
5. Wait for vector embeddings to be created (~30 seconds)

### 3. Start Chatting
- Type your message in the chat input
- The AI responds as the trained persona
- Typing indicator shows "{Name} is typing..."
- Responses use chain-of-thought reasoning with top 3 relevant examples

### 4. Manage Sessions
- View all active sessions in the sidebar
- Switch between different personas
- Delete old sessions to free memory

## 🔬 Explore Your Vectors (Bonus!)

Use the included Jupyter notebook to peek inside the vector embeddings:

```bash
jupyter notebook explore_vectors.ipynb
```

Features:
- View all cached vector stores
- Test similarity searches
- Visualize embeddings in 2D (t-SNE)
- Analyze vector statistics
- Interactive query testing

## 🏗️ Project Structure

```
persona-chatbot/
├── app.py                  # Entry point
├── requirements.txt        # Dependencies
├── .env                    # API keys (create this)
├── explore_vectors.ipynb   # Vector explorer notebook
├── config/
│   └── settings.py        # Configuration & API setup
├── models/
│   └── chat_session.py    # Session management
├── services/
│   └── rag_chatbot.py     # Core chatbot logic & UI
├── utils/
│   ├── embeddings.py      # Vector embeddings & FAISS
│   └── message_parser.py  # WhatsApp parser
└── vector_cache/          # Cached embeddings (auto-created)
```

## 🎯 How It Works

1. **Parse Chat** - Extracts messages from WhatsApp export
2. **Filter Training Data** - Isolates target person's responses only
3. **Create Embeddings** - Converts text to 384-dim vectors using HuggingFace
4. **Build Vector Store** - Indexes embeddings in FAISS for fast retrieval
5. **Cache Vectors** - Saves to disk (MD5 hash-based)
6. **User Query** - Finds top 3 similar examples from training data
7. **Chain-of-Thought** - AI reasons through response step-by-step
8. **Generate Response** - Gemini creates persona-accurate reply

## 🎨 Key Features Explained

### Chain-of-Thought Prompting
The AI uses structured reasoning:
1. What is the person asking/saying?
2. How would {persona} feel about this?
3. What would {persona} typically say?
4. What tone and words would {persona} use?

### Smart Caching
- Vector stores cached with MD5 hash keys
- Models cached with `@st.cache_resource`
- Instant reload on subsequent runs

### Error Handling
- **API Quota Exceeded**: Shows "💤 {Name} is offline"
- **Event Loop Issues**: Auto-creates asyncio loop
- **FAISS Errors**: Graceful fallback to batch processing

## ☁️ Deployment on Streamlit Cloud

### Steps:
1. **Push to GitHub** (already done!)
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin master
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `rebel47/persona-chatbot`
   - Main file: `app.py`

3. **Add Secrets**
   - Click "Advanced settings"
   - Add to secrets:
   ```toml
   GOOGLE_API_KEY = "your_actual_api_key_here"
   ```

4. **Deploy!**
   - Click "Deploy"
   - Your app will be live at `https://your-app.streamlit.app`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Guidelines:
1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## 🐛 Known Issues & Solutions

| Issue | Solution |
|-------|----------|
| `RuntimeError: no running event loop` | Fixed with asyncio loop creation |
| API quota exceeded | Uses local HuggingFace embeddings (unlimited) |
| FAISS import error | Reinstall: `pip install faiss-cpu==1.12.0` |
| Torch warning about `__path__._path` | Harmless, can be ignored |

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Developer

**Mohammad Ayaz Alam**
- GitHub: [@rebel47](https://github.com/rebel47)
- Project: [persona-chatbot](https://github.com/rebel47/persona-chatbot)

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) - Powerful LLM API
- [Streamlit](https://streamlit.io/) - Amazing web framework
- [HuggingFace](https://huggingface.co/) - Open-source ML models
- [LangChain](https://www.langchain.com/) - RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient vector search

## 📊 Performance

- **Embedding Creation**: ~30 seconds for 200 messages
- **Vector Store Cache**: Instant reload after first training
- **Response Time**: 2-5 seconds per message
- **Memory Usage**: ~500MB with loaded models
- **Cache Size**: ~1-5MB per persona

## 🔮 Future Enhancements

- [ ] Voice message analysis
- [ ] Emoji pattern matching
- [ ] Multi-language support
- [ ] Message timestamp analysis
- [ ] Export conversation history
- [ ] Fine-tune LLM on persona data

---

<div align="center">

**Made with ❤️ by [Mohammad Ayaz Alam](https://github.com/rebel47)**

⭐ Star this repo if you find it useful!

</div>
