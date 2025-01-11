import streamlit as st
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import time

@dataclass
class Message:
    """Data class for chat messages"""
    role: str
    content: str
    timestamp: float = time.time()

class CrustdataSupport:
    def __init__(self, groq_api_key: str):
        """Initialize the support chatbot with necessary components"""
        self.groq_api_key = groq_api_key
        self.messages: List[Message] = []
        self.setup_llm()
        self.setup_knowledge_base()
        self.setup_memory()
        
    def setup_llm(self):
        """Configure LLaMA model through Groq"""
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama2-70b-4096",
            temperature=0.2,
            max_tokens=4096
        )
    
    def setup_knowledge_base(self):
        """Initialize and configure the vector store for API documentation"""
        # Initialize text splitter for processing documentation
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store from documentation
        self.knowledge_base = self.create_vector_store()
        
    def create_vector_store(self) -> FAISS:
        """Create and return a vector store from the API documentation"""
        # This is where you'll add the actual API documentation
        api_docs = """
        [Your API documentation content here]
        
        Example endpoints:
        1. Discovery API
        2. Enrichment API
        3. Dataset API
        
        Include detailed documentation about endpoints, parameters, and example responses
        """
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(api_docs)
        
        # Create and return vector store
        return FAISS.from_texts(chunks, self.embeddings)
    
    def setup_memory(self):
        """Initialize conversation memory"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create retrieval chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.knowledge_base.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )
    
    def get_response(self, user_message: str) -> str:
        """Generate a response to the user's message"""
        try:
            # Add system context for better responses
            system_prompt = """You are a helpful and knowledgeable technical support agent for Crustdata's APIs. 
            Your responses should be accurate, clear, and focused on helping users understand and effectively use the APIs.
            Include relevant code examples when appropriate."""
            
            # Get response from chain
            response = self.chain({"question": user_message, "system": system_prompt})
            
            # Add messages to history
            self.messages.append(Message(role="user", content=user_message))
            self.messages.append(Message(role="assistant", content=response["answer"]))
            
            return response["answer"]
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            self.messages.append(Message(role="error", content=error_message))
            return error_message

# Streamlit UI
def create_streamlit_ui():
    st.set_page_config(page_title="Crustdata API Support", layout="wide")
    
    # Initialize session state
    if "support_agent" not in st.session_state:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.session_state.support_agent = CrustdataSupport(groq_api_key)
    
    st.title("Crustdata API Support")
    
    # Chat interface
    st.markdown("### Chat with our API Support Agent")
    
    # Display chat messages
    for message in st.session_state.support_agent.messages:
        if message.role == "user":
            st.markdown(f"**You:** {message.content}")
        elif message.role == "assistant":
            st.markdown(f"**Support Agent:** {message.content}")
        else:
            st.error(message.content)
    
    # User input
    user_input = st.text_input("Ask a question about Crustdata's APIs:", key="user_input")
    
    if st.button("Send"):
        if user_input:
            response = st.session_state.support_agent.get_response(user_input)
            st.experimental_rerun()

if __name__ == "__main__":
    create_streamlit_ui()