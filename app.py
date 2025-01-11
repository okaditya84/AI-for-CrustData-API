import streamlit as st
import groq
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import re
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor

@dataclass
class DocumentChunk:
    """Represents a semantic chunk of documentation"""
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = None

@dataclass
class Message:
    """Represents a chat message with enhanced metadata"""
    role: str
    content: str
    timestamp: float = time.time()
    metadata: Dict = None

class SemanticChunker:
    """Intelligent document chunking without external dependencies"""
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        
    def split_text(self, text: str) -> List[str]:
        """Split text into semantic chunks using rule-based approach"""
        # Split on major section boundaries
        sections = re.split(r'\n\s*#{1,3}\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for section in sections:
            # Split into paragraphs
            paragraphs = section.split('\n\n')
            
            for paragraph in paragraphs:
                if current_length + len(paragraph) > self.max_chunk_size:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(paragraph)
                current_length += len(paragraph)
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

class DocumentStore:
    """In-memory vector store implementation"""
    
    def __init__(self):
        self.documents: List[DocumentChunk] = []
        self.index: Dict[str, List[int]] = {}
        
    def add_documents(self, chunks: List[str]):
        """Add documents to the store with basic indexing"""
        for chunk in chunks:
            doc = DocumentChunk(content=chunk)
            self.documents.append(doc)
            
            # Create inverted index for key terms
            terms = set(re.findall(r'\w+', chunk.lower()))
            for term in terms:
                if term not in self.index:
                    self.index[term] = []
                self.index[term].append(len(self.documents) - 1)
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Search for relevant documents using TF-IDF-like scoring"""
        query_terms = set(re.findall(r'\w+', query.lower()))
        scores = []
        
        for idx, doc in enumerate(self.documents):
            score = 0
            doc_terms = set(re.findall(r'\w+', doc.content.lower()))
            common_terms = query_terms & doc_terms
            
            if common_terms:
                score = len(common_terms) / (len(query_terms) * len(doc_terms)) ** 0.5
            scores.append((idx, score))
        
        # Get top_k documents
        top_docs = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return [self.documents[idx].content for idx, _ in top_docs]

class CrustdataSupport:
    """Advanced Crustdata API support system"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.client = groq.Groq(api_key=groq_api_key)
        self.messages: deque = deque(maxlen=100)  # Limit conversation history
        self.chunker = SemanticChunker()
        self.doc_store = DocumentStore()
        self.load_documentation()
    
    def load_documentation(self):
        """Load and process API documentation"""
        # Discovery and Enrichment API Documentation
        discovery_api_docs = """
        # Crustdata Discovery And Enrichment API
        
        ## Overview
        The Discovery API allows you to search and discover companies.
        The Enrichment API provides detailed company information.
        
        ## Endpoints
        
        ### GET /api/v1/companies/search
        Search for companies using various filters.
        
        Parameters:
        - query: Search query string
        - filters: Optional filtering criteria
        
        ### GET /api/v1/companies/{company_id}/enrich
        Get enriched data for a specific company.
        
        Parameters:
        - company_id: Unique identifier for the company
        """
        
        # Dataset API Documentation
        dataset_api_docs = """
        # Crustdata Dataset API
        
        ## Overview
        Access and manage datasets through our comprehensive API.
        
        ## Endpoints
        
        ### GET /api/v1/datasets
        List available datasets.
        
        ### GET /api/v1/datasets/{dataset_id}
        Get details of a specific dataset.
        
        Parameters:
        - dataset_id: Unique identifier for the dataset
        """
        
        # Process and store documentation
        all_docs = [discovery_api_docs, dataset_api_docs]
        for doc in all_docs:
            chunks = self.chunker.split_text(doc)
            self.doc_store.add_documents(chunks)
    
    def get_response(self, user_message: str) -> str:
        """Generate response using context-aware processing"""
        try:
            # Get relevant documentation chunks
            relevant_docs = self.doc_store.search(user_message)
            
            # Construct prompt with context
            context = "\n\n".join(relevant_docs)
            prompt = f"""You are a specialized technical support agent for Crustdata's APIs.
            Use the following documentation context to answer the user's question:
            
            {context}
            
            User Question: {user_message}
            
            Provide a clear, accurate response with specific examples when relevant."""
            
            # Get response from Groq
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.4,
                max_tokens=4098
            )
            
            response = chat_completion.choices[0].message.content
            
            # Store messages
            self.messages.append(Message(role="user", content=user_message))
            self.messages.append(Message(role="assistant", content=response))
            
            return response
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            self.messages.append(Message(role="error", content=error_msg))
            return error_msg

def create_streamlit_ui():
    """Create the Streamlit user interface"""
    st.set_page_config(
        page_title="Crustdata API Support",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "support_agent" not in st.session_state:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.session_state.support_agent = CrustdataSupport(groq_api_key)
    
    st.title("Crustdata API Support")
    
    # Sidebar with API documentation overview
    with st.sidebar:
        st.header("API Documentation")
        st.markdown("""
        ### Available APIs
        - Discovery API
        - Enrichment API
        - Dataset API
        
        Ask questions about endpoints, parameters, and usage!
        """)
    
    # Main chat interface
    st.markdown("### Chat with our API Support Agent")
    
    # Message container with custom styling
    message_container = st.container()
    
    with message_container:
        for msg in st.session_state.support_agent.messages:
            if msg.role == "user":
                st.markdown(f"🧑 **You:** {msg.content}")
            elif msg.role == "assistant":
                st.markdown(f"🤖 **Support Agent:** {msg.content}")
            else:
                st.error(msg.content)
    
    # User input area
    st.markdown("---")
    user_input = st.text_input(
        "Ask a question about Crustdata's APIs:",
        key="user_input",
        placeholder="e.g., How do I use the company search endpoint?"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("Send", use_container_width=True):
            if user_input:
                response = st.session_state.support_agent.get_response(user_input)
                st.rerun()
    
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.support_agent.messages.clear()
            st.rerun()

if __name__ == "__main__":
    create_streamlit_ui()