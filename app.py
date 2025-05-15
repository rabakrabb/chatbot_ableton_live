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

# --- CSS styling ---
st.markdown("""
<style>
/* Hela body och appens root */
body, .css-18e3th9 {
    margin: 0; padding: 0; min-height: 100vh;
    background-color: var(--backgroundColor) !important;
    color: var(--textColor) !important;
    font-family: var(--font-family, "sans-serif") !important;
}

/* Container för appens innehåll */
.app-container {
    max-width: 800px;
    margin: 40px auto;
    background-color: var(--secondaryBackgroundColor) !important;
    color: var(--textColor) !important;
    padding: 30px 40px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    font-family: var(--font-family, "sans-serif") !important;
}

/* Rubriker */
h1, h2, h3 {
    color: var(--primaryColor) !important;
    font-weight: 700 !important;
}

/* Textinput */
.stTextInput > div > div > input {
    background-color: white !important;
    color: var(--textColor) !important;
    border: 2px solid var(--primaryColor) !important;
    border-radius: 6px !important;
    padding: 8px !important;
    font-family: var(--font-family, "sans-serif") !important;
}

/* Knappar */
div.stButton > button {
    background-color: var(--primaryColor) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease !important;
    font-family: var(--font-family, "sans-serif") !important;
}
div.stButton > button:hover {
    background-color: #e65b50 !important;
}

/* Länkar */
a {
    color: var(--primaryColor) !important;
    font-family: var(--font-family, "sans-serif") !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def initialize_rag(jsonl_path: str = "chunks.jsonl"):
    chunks: List[Dict] = load_chunks(jsonl_path)
    contents = [chunk["content"] for chunk in chunks]
    embeddings = create_embeddings(contents)
    return chunks, embeddings

chunks, embeddings = initialize_rag()

# --- Meny ---
menu = ["Chatbot", "About the app"]
st.sidebar.markdown("### Navigation")
selection = st.sidebar.radio("Choose page", menu, index=0)  # default = Chatbot

st.markdown('<div class="app-container">', unsafe_allow_html=True)

if selection == "Chatbot":
    st.title("The Ableton Live 12 MIDI RAG-Bot")
    query = st.text_input("Ask your question:")

    if query:
        query_emb = create_embeddings([query])[0]
        texts = [chunk["content"] for chunk in chunks]
        top_texts = semantic_search(query_emb, texts, embeddings, top_k=5)
        context = "\n\n".join(top_texts)
        answer = generate_response(query, context)
        st.markdown("### Answer:")
        st.write(answer)

elif selection == "About the app":
    st.title("About the app")
    st.write("""
    This chatbot is built using semantic search and retrieval-augmented generation (RAG) to answer questions about Ableton Live 12 MIDI.

    It uses precomputed embeddings of Ableton Live 12 manual chunks and Google's AI technology to generate context-aware answers.

    Created by Martin Blomqvist during the Data Scientist program at EC Utbildning 2025.

    For more information:
    - [GitHub](https://github.com/rabakrabb)
    - [LinkedIn](https://www.linkedin.com/in/martin-blomqvist)
    """)

st.markdown('</div>', unsafe_allow_html=True)
