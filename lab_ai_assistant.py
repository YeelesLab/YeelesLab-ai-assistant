import os
import tempfile
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load API key from Streamlit secrets or environment
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ Missing OpenAI API key. Add it to `.env` or Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="Yeeles Lab Assistant", page_icon="🧬")
st.title("🧬 Yeeles Lab AI Assistant")
st.write("Upload a research paper and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("🔍 Reading PDF..."):
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)

        # Embedding and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Retrieval-based QA
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=openai_api_key),
            retriever=retriever,
            chain_type="stuff"
        )

        # Ask a question
        question = st.text_input("Ask a question about the paper:")

        if question:
            with st.spinner("🤖 Generating answer..."):
                answer = qa_chain.run(question)
                st.success("✅ Answer:")
                st.write(answer)
