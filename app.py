import streamlit as st
import json
import os
from typing import List, Dict
from dotenv import load_dotenv

# ---- dina egna moduler ----
from semantic_search import semantic_search
from llm_utils import generate_response
from rag_utils import create_embeddings, load_chunks

# ---- sidinställningar ----
st.set_page_config(
    page_title="The Ableton Live 12 MIDI RAG-Bot",
    layout="centered",  # centrera appen smidigt
    initial_sidebar_state="auto",
    page_icon="🎹",
)

load_dotenv()

# FIXERAD BREDD på app-rutan + färgtema turkost/korall
st.markdown(
    """
    <style>
    /* Fixerad bredd och centrerad app */
    section.main {
        max-width: 800px !important;
        width: 800px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        background-color: #e0f7f9 !important;  /* mjuk turkos bakgrund */
        color: #003a3f !important;               /* mörk turkos text */
    }

    div.block-container {
        background-color: #e0f7f9 !important;
        color: #003a3f !important;
    }

    /* Titlar */
    h1, h2, h3 {
        color: #ff6f61 !important;  /* varm korall/orange */
        font-weight: 700 !important;
    }

    /* Text input */
    .stTextInput > div > div > input {
        border: 2px solid #00bcd4 !important;  /* klar turkos kant */
        border-radius: 6px !important;
        padding: 8px !important;
        color: #003a3f !important;
        background-color: white !important;
    }

    /* Knappar */
    div.stButton > button {
        background-color: #ff6f61 !important;  /* korall */
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #e65b50 !important; /* mörkare korall vid hover */
    }

    /* Länkfärg */
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
