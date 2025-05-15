import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from chunking import chunk_text_from_file
from rag_utils import create_embeddings, load_chunks
from semantic_search import semantic_search
from llm_utils import generate_response

load_dotenv()  # only needed locally; in Streamlit Cloud use st.secrets

# configure Google GenAI
API_KEY = os.getenv("API_KEY") or st.secrets["API_KEY"]
genai.configure(api_key=API_KEY)

@st.cache_resource(show_spinner=False)
def initialize_rag(
    raw_txt: str = "data/extracted_midi_chapters.txt",
    jsonl_path: str = "chunks.jsonl",
):
    # 1. Ensure we have chunked JSONL
    if not os.path.isfile(jsonl_path):
        chunk_text_from_file(raw_txt, jsonl_path)

    # 2. Load chunks
    chunks = load_chunks(jsonl_path)
    contents = [c["content"] for c in chunks]

    # 3. Create embeddings once
    embeddings = create_embeddings(contents)
    return chunks, embeddings

chunks, embeddings = initialize_rag()

st.title("The Ableton Live 12 MIDI Beginners Chatbot (RAG-Based)")

query = st.text_input("Ask your question here:")

if query:
    # embed query
    q_emb = create_embeddings([query])[0]
    # semantic search
    texts = [c["content"] for c in chunks]
    relevant = semantic_search(q_emb, texts, embeddings)
    context = "\n\n".join(relevant)
    # generate answer
    answer = generate_response(query, context)
    st.markdown("### Answer:")
    st.write(answer)
