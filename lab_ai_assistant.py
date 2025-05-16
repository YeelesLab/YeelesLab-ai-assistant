# lab_ai_assistant.py

import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile
from langchain.vectorstores import FAISS


# Set OpenAI API key (or use st.secrets for Streamlit Cloud)
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Lab AI Assistant", layout="wide")
st.title("🧠 Lab AI Research Assistant")

uploaded_files = st.file_uploader("Upload one or more research papers (PDFs)", type=["pdf"], accept_multiple_files=True)
query = st.text_input("Ask a question about the uploaded papers:")

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        all_texts = []

        for uploaded_file in uploaded_files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                loader = PyPDFLoader(tmp_file.name)
                pages = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                texts = text_splitter.split_documents(pages)
                all_texts.extend(texts)

        # Embed and store in Chroma
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(all_texts, embeddings)

        # Retrieval-based QA chain
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        if query:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(query)
                st.subheader("Answer:")
                st.write(answer)
else:
    st.info("Please upload at least one PDF to begin.")
