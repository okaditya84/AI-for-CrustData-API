import os
import re
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict
import json

# Constants
TEXT_FILES_FOLDER = "text_files"
VECTOR_STORE_PATH = "vectorstore"
MEMORY_KEY = "chat_history"
MODEL_NAME = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MEMORY_WINDOW = 5

class CrustdataBot:
    def __init__(self):
        self.groq_api_key = os.environ['GROQ_API_KEY']
        self.memory = self._initialize_memory()
        self.retriever = self._initialize_retriever()
        self.llm = self._initialize_llm()
        self.conversation_chain = self._create_conversation_chain()
    
    def _initialize_memory(self) -> ConversationBufferMemory:
        return ConversationBufferMemory(
            k=MEMORY_WINDOW,
            memory_key=MEMORY_KEY,
            return_messages=True,
            input_key="human_input"
        )

    def _initialize_retriever(self):
        try:
            if os.path.exists(VECTOR_STORE_PATH):
                vectorstore = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                    allow_dangerous_deserialization=True
                )
            else:
                docs = self._load_and_process_documents()
                if not docs:
                    raise ValueError("No documents found for processing")
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local(VECTOR_STORE_PATH)
            
            return vectorstore.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            st.error(f"Error initializing retriever: {str(e)}")
            return None

    def _initialize_llm(self) -> ChatGroq:
        return ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=MODEL_NAME,
            temperature=0.3,
            max_tokens=1000,
            top_p=0.9
        )

    def _create_conversation_chain(self) -> LLMChain:
        system_prompt = """You are an expert Crustdata API support specialist. Your responses should be:
1. Strictly based on the Crustdata API documentation, avoiding any information not present in the documentation.
2. Technical and precise, focusing on API endpoints, parameters, and usage.
3. Include relevant code examples exactly as provided in the API documentation.
4. Explain any limitations or requirements clearly.
5. Suggest workarounds for common issues based on the documentation.
6. Use consistent formatting for API endpoints and parameters.
7. If any API request example is provided, validate it before returning to the user. If there's an error, suggest a fix based on logs or known usage.

If you're unsure about any details, acknowledge the uncertainty and suggest where to find more information in the documentation."""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            HumanMessagePromptTemplate.from_template(
                "Question: {human_input}\n\nRelevant Documentation:\n{context}\n\n"
                "Provide a technical answer focusing on Crustdata's API usage. Validate any API request snippet."
            ),
        ])

        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=self.memory,
        )

    def _load_and_process_documents(self) -> List[Document]:
        docs = []
        for filename in os.listdir(TEXT_FILES_FOLDER):
            if filename.endswith(".txt"):
                with open(os.path.join(TEXT_FILES_FOLDER, filename), "r", encoding="utf-8") as file:
                    text = file.read()
                    docs.append(
                        Document(page_content=text, metadata={"source": filename, "type": "api_documentation"})
                    )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_documents(docs)

    def get_response(self, question: str) -> str:
        try:
            relevant_docs = self.retriever.get_relevant_documents(question)
            context = self._process_relevant_documents(relevant_docs)
            response = self.conversation_chain.predict(human_input=question, context=context)
            response = self._validate_api_requests_in_response(response)
            return self._format_response(response)
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."

    def _process_relevant_documents(self, docs: List[Document]) -> str:
        processed_chunks = []
        for doc in docs:
            chunk = doc.page_content.strip()
            if chunk:
                processed_chunks.append(chunk)
        return "\n\n---\n\n".join(processed_chunks)

    def _format_response(self, response: str) -> str:
        response = response.replace("```curl", "```bash")
        response = response.replace("api.crustdata.com", "`api.crustdata.com`")
        return response

    def _validate_api_requests_in_response(self, response: str) -> str:
        curl_pattern = re.compile(r"```bash\s*(curl[^\n]+)\s*```", re.IGNORECASE | re.DOTALL)
        matches = curl_pattern.findall(response)
        fixed_response = response
        for match in matches:
            if "api.crustdata.com" not in match:
                correction = match.replace("curl ", "curl https://api.crustdata.com ")
                fixed_snippet = f"```bash\n{correction}\n```"
                fixed_response = fixed_response.replace(f"```bash\n{match}\n```", fixed_snippet)
        return fixed_response

def main():
    st.set_page_config(page_title="Crustdata Chat Support", layout="centered")
    st.title("Crustdata API Support - Chat Mode")

    bot = CrustdataBot()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask something about Crustdata's APIs...")
    if user_input:
        response = bot.get_response(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", response))

    for role, content in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

if __name__ == "__main__":
    main()
