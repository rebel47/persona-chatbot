from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
import os
import logging
import pickle
import hashlib
from pathlib import Path
import streamlit as st

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    from config.settings import GEMINI_API_KEY
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    GEMINI_API_KEY = None

logger = logging.getLogger(__name__)


@st.cache_resource
def load_huggingface_model():
    """Load and cache the HuggingFace model to avoid reloading."""
    logger.info("Loading HuggingFace model (will be cached)...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


class GoogleGenAIEmbeddings(Embeddings):
    """Custom embeddings class using google-genai SDK."""
    
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        """Initialize the embeddings with google-genai client."""
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.batch_size = 100  # Google API limit
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with batching support."""
        try:
            all_embeddings = []
            
            # Process in batches of 100
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                logger.info(f"Embedding batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1} ({len(batch)} texts)")
                
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch
                )
                
                batch_embeddings = [embedding.values for embedding in result.embeddings]
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Successfully embedded {len(all_embeddings)} documents")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=[text]
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise


class EmbeddingManager:
    """
    Manages document embeddings using HuggingFace (local) or Google's Generative AI (API).
    Prefers HuggingFace to avoid API quota issues.
    """
    
    def __init__(self, use_huggingface: bool = True):
        """
        Initialize the embedding manager.
        
        Args:
            use_huggingface (bool): If True, use HuggingFace embeddings (local, free, unlimited).
                                   If False or unavailable, fall back to Google GenAI.
        """
        try:
            # Try HuggingFace first (local, no API limits!)
            if use_huggingface and HUGGINGFACE_AVAILABLE:
                logger.info("Initializing HuggingFace embeddings (local, unlimited)...")
                # Use cached model
                self.embeddings = load_huggingface_model()
                self.embedding_type = "huggingface"
                logger.info("✅ Using HuggingFace embeddings - No API limits!")
                
            # Fall back to Google GenAI if HuggingFace is not available
            elif GOOGLE_GENAI_AVAILABLE and GEMINI_API_KEY:
                logger.info("Initializing Google GenAI embeddings (API-based)...")
                self.embeddings = GoogleGenAIEmbeddings(
                    api_key=GEMINI_API_KEY,
                    model="gemini-embedding-001"
                )
                self.embedding_type = "google"
                logger.info("✅ Using Google GenAI embeddings")
                
            else:
                raise RuntimeError(
                    "No embedding provider available! "
                    "Please install langchain-huggingface or configure GOOGLE_API_KEY"
                )
            
            # Create cache directory for vector stores
            self.cache_dir = Path("vector_cache")
            self.cache_dir.mkdir(exist_ok=True)
            
            logger.info(f"Successfully initialized embedding manager ({self.embedding_type})")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
    
    def _get_cache_key(self, documents: List[Dict[str, str]]) -> str:
        """
        Generate a unique cache key based on document content.
        
        Args:
            documents (List[Dict[str, str]]): List of documents
            
        Returns:
            str: MD5 hash of document content
        """
        content = "".join([f"{doc['prompt']}{doc['completion']}" for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cached_vector_store(self, cache_key: str) -> Optional[FAISS]:
        """
        Load a cached vector store if it exists.
        
        Args:
            cache_key (str): The cache key
            
        Returns:
            Optional[FAISS]: Cached vector store or None
        """
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                logger.info(f"Loading cached vector store: {cache_key}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
                return None
        return None
    
    def _save_vector_store_cache(self, cache_key: str, vector_store: FAISS) -> None:
        """
        Save a vector store to cache.
        
        Args:
            cache_key (str): The cache key
            vector_store (FAISS): The vector store to cache
        """
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(vector_store, f)
            logger.info(f"Saved vector store to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")

    def create_vector_store(self, documents: List[Dict[str, str]]) -> FAISS:
        """
        Create a FAISS vector store from formatted documents.
        Uses caching to avoid re-embedding the same content.
        
        Args:
            documents (List[Dict[str, str]]): List of documents with 'prompt' and 'completion'
            
        Returns:
            FAISS: Initialized FAISS vector store
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(documents)
            cached_store = self._load_cached_vector_store(cache_key)
            
            if cached_store:
                logger.info("Using cached vector store - no API calls needed!")
                return cached_store
            
            # Create new vector store if not cached
            logger.info(f"Creating new vector store with {len(documents)} documents...")
            docs = [
                Document(page_content=f"{item['prompt']} {item['completion']}")
                for item in documents
            ]
            
            vector_store = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            # Cache the vector store
            self._save_vector_store_cache(cache_key, vector_store)
            
            logger.info(f"Successfully created vector store with {len(docs)} documents")
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise

    def get_similar_documents(self, query: str, vector_store: FAISS, k: int = 4) -> List[Document]:
        """
        Retrieve similar documents from the vector store.
        
        Args:
            query (str): The search query
            vector_store (FAISS): The FAISS vector store
            k (int): Number of similar documents to retrieve
            
        Returns:
            List[Document]: List of similar documents
        """
        try:
            docs = vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise