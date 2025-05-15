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
    initial_sidebar_state="auto",
    page_icon="🎹",
)

load_dotenv()

st.markdown(
    """
    <style>
    /* Sida bakgrund: mörkturkos */
    .css-1d391kg {
        background-color: #004d4d !important;  /* mörk turkos */
        min-height: 100vh;
    }

    /* App-ruta: ljus turkos med rundade hörn och skugga */
    section.main {
        max-width: 800px !important;
        width: 800px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        background-color: #e0f7f9 !important;  /* ljus turkos */
        color: #003a3f !important;              /* mörk turkos text */
        padding: 30px 40px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        position: relative;
        z-index: 10;
    }

    /* Rubriker: korall */
    h1, h2, h3 {
        color: #ff6f61 !important;  /* korall */
        font-weight: 700 !important;
    }

    /* Textinput: vit bakgrund med turkos kant */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: #003a3f !important;
        border: 2px solid #00bcd4 !important;  /* klar turkos kant */
        border-radius: 6px !important;
        padding: 8px !important;
    }

    /* Knappar: korall bakgrund och vit text */
    div.stButton > button {
        background-color: #ff6f61 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #e65b50 !important;  /* lite mörkare korall */
    }

    /* Länkar: korall */
    a {
        color: #ff6f61 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def initialize_rag(jsonl_path: str = "chunks.jsonl"):
    chunks: List[Dict] = load_chunks(jsonl_path)
    contents = [chunk["content"] for chunk in chunks]
    embeddings = create_embeddings(contents)
    return chunks, embeddings

chunks, embeddings = initialize_rag()

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
