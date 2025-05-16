import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv

# Egna moduler
from semantic_search import semantic_search
from llm_utils import generate_response
from rag_utils import create_embeddings, load_chunks

st.set_page_config(
    page_title="The Ableton Live 12 MIDI RAG-Bot",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🎹",
)

load_dotenv()

# --- CSS styling med textfärgs-justeringar ---
st.markdown("""
<style>
:root {
    --primaryColor: #FF6F61;             /* Korall */
    --backgroundColor: #004D4D;          /* Mörk turkos */
    --secondaryBackgroundColor: #006666; /* Mörkare sekundär */
    --font-family: "Arial, sans-serif";
}

/* Bakgrund och font */
body, main {
    margin: 0; padding: 0; min-height: 100vh;
    background-color: var(--backgroundColor) !important;
    color: white !important;                     /* All brödtext blir vit */
    font-family: var(--font-family) !important;
}

/* Styla Streamlits container som 'app-ruta' */
section.main, div.block-container {
    max-width: 800px !important;
    margin: 80px auto 40px auto !important;
    background-color: var(--secondaryBackgroundColor) !important;
    padding: 30px 40px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
}

/* Rubriker */
h1, h2, h3 {
    color: var(--primaryColor) !important;
    font-weight: 700 !important;
    margin-top: 0 !important;
}

/* Textinput */
.stTextInput > div > div > input {
    width: 100% !important;
    background-color: white !important;
    color: black !important;                     /* Input-text svart */
    border: 2px solid var(--primaryColor) !important;
    border-radius: 6px !important;
    padding: 8px !important;
    font-family: var(--font-family) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #888 !important; /* placeholder lätt grå */
}

/* Knappar */
div.stButton > button {
    background-color: var(--primaryColor) !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease !important;
    font-family: var(--font-family) !important;
}
div.stButton > button:hover {
    background-color: #e65b50 !important;
}

/* Menyfärg */
[data-testid="stSidebar"] {
    background-color: var(--secondaryBackgroundColor) !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def initialize_rag(jsonl_path: str = "chunks.jsonl"):
    chunks = load_chunks(jsonl_path)
    chunks = [c for c in chunks if c.get("content", "").strip()]
    embeddings = create_embeddings([c["content"] for c in chunks])
    return chunks, embeddings

chunks, embeddings = initialize_rag()

# --- Meny ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Chatbot", "About the app"], index=0)

if page == "Chatbot":
    st.title("The Ableton Live 12 MIDI RAG-Bot")
    query = st.text_input("Ask your question:")

    if query:
        query_emb = create_embeddings([query])[0]
        texts = [c["content"] for c in chunks]
        top_texts = semantic_search(query_emb, texts, embeddings, top_k=5)
        context = "\n\n".join(top_texts)
        answer = generate_response(query, context)
        st.markdown("### Answer:")
        st.write(answer)

else:
    st.title("About the app")
    st.write("""
This chatbot is built using semantic search and retrieval-augmented generation (RAG) to answer questions about Ableton Live 12 MIDI.

It uses precomputed embeddings of Ableton Live 12 manual chunks and Google's AI technology to generate context-aware answers.

Created by Martin Blomqvist during the Data Scientist program at EC Utbildning 2025.

For more information:
- [GitHub](https://github.com/rabakrabb)
- [LinkedIn](https://www.linkedin.com/in/martin-blomqvist)
""")
