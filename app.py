import streamlit as st
import json
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

# Bakgrund som täcker hela sidan - vi kan lägga i markdown en div som ligger bakom allt
st.markdown(
    """
    <style>
    .app-background {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background-color: #004d4d;  /* mörk turkos */
        z-index: 0;
    }
    </style>
    <div class="app-background"></div>
    """,
    unsafe_allow_html=True,
)

# Skapa en container som vi kan styla med inline CSS
with st.container():
    st.markdown(
        """
        <style>
        .app-container {
            max-width: 800px;
            margin: 60px auto 40px auto;
            padding: 30px 40px;
            border-radius: 12px;
            background-color: #e0f7f9;  /* ljus turkos */
            color: #003a3f;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            position: relative;
            z-index: 10;
        }
        h1, h2, h3 {
            color: #ff6f61;
            font-weight: 700;
        }
        .stTextInput > div > div > input {
            border: 2px solid #00bcd4;
            border-radius: 6px;
            padding: 8px;
            color: #003a3f;
            background-color: white;
        }
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
        a {
            color: #ff6f61;
        }
        </style>
        <div class="app-container">
        """,
        unsafe_allow_html=True,
    )

    # Själva appinnehållet under "app-container"
    st.title("The Ableton Live 12 MIDI RAG-Bot")
    query = st.text_input("Ask your question:")

    if query:
        chunks, embeddings = initialize_rag()
        query_emb = create_embeddings([query])[0]
        texts = [chunk["content"] for chunk in chunks]
        top_texts = semantic_search(query_emb, texts, embeddings, top_k=5)
        context = "\n\n".join(top_texts)
        answer = generate_response(query, context)
        st.markdown("### Answer:")
        st.write(answer)

    # Stäng div
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def initialize_rag(jsonl_path: str = "chunks.jsonl"):
    chunks: List[Dict] = load_chunks(jsonl_path)
    contents = [chunk["content"] for chunk in chunks]
    embeddings = create_embeddings(contents)
    return chunks, embeddings
