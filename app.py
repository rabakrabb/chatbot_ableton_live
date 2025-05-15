import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv

# ---- dina egna moduler ----
from semantic_search import semantic_search
from llm_utils import generate_response
from rag_utils import create_embeddings, load_chunks

# ---- sidinställningar ----
st.set_page_config(
    page_title="The Ableton Live 12 MIDI RAG-Bot",
    layout="centered",
    initial_sidebar_state="auto",
    page_icon="🎹",
)

load_dotenv()

# Enkelt mörkt tema med ljus text
st.markdown(
    """
    <style>
    /* Bakgrund och textfärg för hela sidan */
    .main {
        background-color: #004d4d;  /* mörk turkos */
        color: #e0f7f9;             /* ljus turkos */
        padding: 2rem 3rem;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Titlar */
    h1, h2, h3 {
        color: #ff6f61;  /* korall */
        font-weight: 700;
    }

    /* Textinput */
    .stTextInput > div > div > input {
        background-color: #006666; /* mörkare turkos */
        color: #e0f7f9;
        border: 2px solid #00bcd4;
        border-radius: 6px;
        padding: 8px;
    }

    /* Knappar */
    div.stButton > button {
        background-color: #ff6f61;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #e65b50;
    }

    /* Länkfärg */
    a {
        color: #ff6f61;
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
