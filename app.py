# ==========================================
# 1) Imports
# ==========================================
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# 2) Load API Key
# ==========================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("⚠ GOOGLE_API_KEY not found. Add it in .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ==========================================
# 3) Streamlit Page Setup
# ==========================================
st.set_page_config(
    page_title="AI-Powered HR Assistant (Gemini + HF Embeddings)",
    layout="wide"
)

st.title("🤖 AI-Powered HR Assistant")
st.caption("Upload an HR policy PDF and ask questions. Powered by Gemini + HuggingFace")


# ==========================================
# 4) Upload PDF
# ==========================================
uploaded_file = st.file_uploader("📄 Upload HR Policy PDF", type=["pdf"])

if uploaded_file:
    pdf_path = "uploaded_hr_policy.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF uploaded successfully")


    # ==========================================
    # 5) Load + Split PDF
    # ==========================================
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)


    # ==========================================
    # 6) Embeddings + Vector DB
    # ==========================================
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(docs, embeddings)


    # ==========================================
    # 7) Gemini LLM + Retrieval QA
    # ==========================================
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1
    )

    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )


    # ==========================================
    # 8) Ask Question
    # ==========================================
    query = st.text_input("❓ Ask a question about the document")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)

        st.write("### ✅ Answer:")
        st.write(response)
