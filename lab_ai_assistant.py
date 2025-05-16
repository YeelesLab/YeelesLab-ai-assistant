import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load your OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Missing OpenAI API key. Please add it to Streamlit secrets.")
    st.stop()

# UI
st.title("🧬 Yeeles Lab AI Assistant")
uploaded_file = st.file_uploader("Upload a PDF paper", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        loader = PyPDFLoader(uploaded_file.name)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(pages)

        # Embeddings & vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)

        # QA chain
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever
        )

    question = st.text_input("Ask a question about this paper:")

    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
            st.markdown(f"**Answer:** {answer}")
