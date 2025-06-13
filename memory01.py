import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain.chains import ConversationalRetrievalChain

# -------------------------------
# 1. HARDCODE GOOGLE API KEY
# -------------------------------
import os
os.environ["GOOGLE_API_KEY"] = "your_google_api_key_here"  # Replace with your actual key

# -------------------------------
# 2. SETUP STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="RAG + Memory with Gemini", layout="wide")
st.title("📚 RAG with Memory using LangChain + Gemini")
st.markdown("This app uses Google Gemini + HuggingFace Embeddings + FAISS for RAG and memory.")

# -------------------------------
# 3. LOAD PDF DOCUMENT
# -------------------------------
@st.cache_resource
def load_and_index_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Replace with your PDF path
pdf_path = "your_pdf_file.pdf"  # Ensure this file is in the same directory
retriever = load_and_index_pdf(pdf_path)

# -------------------------------
# 4. DEFINE MEMORY
# -------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------------------
# 5. SETUP PROMPT TEMPLATE
# -------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer the user's question."),
    ("human", "Context:\n{context}\n\nChat History:\n{chat_history}\n\nQuestion: {question}")
])

# -------------------------------
# 6. INITIALIZE GOOGLE GEMINI
# -------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# -------------------------------
# 7. BUILD RAG CHAIN USING Runnable
# -------------------------------
# Context -> Prompt -> LLM -> Output
def format_inputs(inputs):
    return {
        "question": inputs["question"],
        "context": "\n\n".join([doc.page_content for doc in inputs["context"]]),
        "chat_history": "\n".join([f"{msg.type}: {msg.content}" for msg in memory.chat_memory.messages])
    }

chain = (
    RunnableMap({"question": lambda x: x["question"], "context": lambda x: retriever.get_relevant_documents(x["question"])})
    | RunnableLambda(format_inputs)
    | prompt
    | llm
)

# -------------------------------
# 8. STREAMLIT INTERFACE
# -------------------------------
user_input = st.text_input("Ask a question in English:")

if user_input:
    with st.spinner("Thinking..."):
        # Run chain
        response = chain.invoke({"question": user_input})
        # Save message to memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response.content)
        # Show response
        st.markdown("### 💬 Answer")
        st.write(response.content)

    # Display chat history
    st.markdown("### 🧠 Chat Memory")
    for msg in memory.chat_memory.messages:
        role = "🧑 You" if msg.type == "human" else "🤖 AI"
        st.markdown(f"**{role}:** {msg.content}")
