import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


TEXT_FILES_FOLDER = "text_files"  # Folder containing text files
VECTOR_STORE_PATH = "vectorstore"  # Path to save/load the vector store


def main():
    """
    This function initializes the Groq chatbot and integrates text file-based Q&A.
    """
    # Set up Groq API key
    groq_api_key = os.environ['GROQ_API_KEY']

    # Streamlit app interface
    st.title("Customer Support Chatbot")
    st.write("Interact with the Crustdata customer support chatbot!")

    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:", value="You are a helpful customer support agent.")
    model = st.sidebar.selectbox('Choose a model', ['llama-3.3-70b-versatile'])
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(
    k=conversational_memory_length,
    memory_key="chat_history",
    return_messages=True,
    input_key="human_input"  # Specify the input key for memory
)

    user_question = st.text_input("Ask a question:")
    submit_button = st.button("Submit")

    # Session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Load or create the vector store
    retriever = load_or_create_retriever()
    if retriever is None:
        st.error("No data found in the text files folder. Please add text files and try again.")
        return
    st.sidebar.success("Vector store loaded successfully!")

    # Initialize Groq LangChain chat object
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    if user_question and submit_button:
        # Use retriever for context
        related_docs = retriever.get_relevant_documents(user_question)
        context = "\n".join([doc.page_content for doc in related_docs])

        # Construct prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}\nContext:\n{context}"),
        ])

        # Create conversation chain
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        # Get response
        response = conversation.predict(human_input=user_question, context=context)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)


def load_or_create_retriever():
    """
    Load the vector store if it exists, otherwise create it from text files.
    """
    if os.path.exists(VECTOR_STORE_PATH):
        # Load existing vector store
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
        return vectorstore.as_retriever()
    else:
        # Create vector store from text files
        docs = load_text_files(TEXT_FILES_FOLDER)
        if docs:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(VECTOR_STORE_PATH)
            return vectorstore.as_retriever()
        else:
            return None


def load_text_files(folder_path):
    """
    Load and process text files into LangChain Document objects.
    """
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                docs.append(Document(page_content=text, metadata={"source": filename}))

    # Split large documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


if __name__ == "__main__":
    main()
