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
st.set_page_config(page_title="The Ableton Live 12 MIDI RAG-Bot", layout="wide")
load_dotenv()

# Lägg till CSS för att begränsa bredden på huvudinnehållet
st.markdown(
    """
    <style>
    .main-content {
        max-width: 800px;  /* Justera detta värde för önskad maxbredd */
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def initialize_rag(jsonl_path: str = "chunks.jsonl"):
    """
    1) Läser in chunk‐metadata (utan embeddings) från JSONL.
    2) Skapar embedding för varje chunk.content.
    """
    # Läs in chunk‐poster
    chunks: List[Dict] = load_chunks(jsonl_path)
    # Extrahera bara innehållet
    contents = [chunk["content"] for chunk in chunks]
    # Skapa embeddingar (enligt rag_utils.create_embeddings)
    embeddings = create_embeddings(contents)
    return chunks, embeddings

chunks, embeddings = initialize_rag()

# Använd en div med klassen "main-content" för att begränsa bredden
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.title("The Ableton Live 12 MIDI RAG-Bot")
query = st.text_input("Ask your question:")

if query:
    # 1) embedda själva frågan
    query_emb = create_embeddings([query])[0]
    # 2) semantisk sökning bland dina chunk‐embeddings
    texts = [chunk["content"] for chunk in chunks]
    top_texts = semantic_search(query_emb, texts, embeddings, top_k=5)
    context = "\n\n".join(top_texts)
    # 3) generera svar
    answer = generate_response(query, context)
    st.markdown("### Answer:")
    st.write(answer)
st.markdown('</div>', unsafe_allow_html=True)  # Avsluta div:en
