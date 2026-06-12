import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from groq import Groq

from config import CHROMA_PATH, COLLECTION


# ══════════════════════════════════════════
# CACHED RESOURCE INITIALISATION
# ══════════════════════════════════════════

@st.cache_resource
def init_resources():
    """Return (embeddings, llm, groq_audio_client)."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    try:
        groq_audio_client = Groq()
    except Exception as e:
        st.error(f"Failed to initialize Groq Audio Client. Error: {e}")
        groq_audio_client = None
    return embeddings, llm, groq_audio_client


@st.cache_resource
def init_vector_store():
    """Return the ChromaDB vector store (depends on embeddings being ready first)."""
    embeddings, _, _ = init_resources()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )