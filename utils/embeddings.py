from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os
import logging
from config.settings import GEMINI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages document embeddings using Google's Generative AI and FAISS.
    """
    
    def __init__(self):
        """Initialize the embedding manager with Google's Generative AI embeddings."""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GEMINI_API_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    def create_vector_store(self, documents: List[Dict[str, str]]) -> FAISS:
        """
        Create a FAISS vector store from formatted documents.
        
        Args:
            documents (List[Dict[str, str]]): List of documents with 'prompt' and 'completion'
            
        Returns:
            FAISS: Initialized FAISS vector store
        """
        try:
            docs = [
                Document(page_content=f"{item['prompt']} {item['completion']}")
                for item in documents
            ]
            
            vector_store = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
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