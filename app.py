import streamlit as st
import json
from rag_utils import create_embeddings, load_chunks
from llm_utils import generate_response
from semantic_search import semantic_search
from dotenv import load_dotenv
import os
import google.genai as genai

load_dotenv()


@st.cache_resource(show_spinner=False)
def initialize_rag(jsonl_path: str = "chunks.jsonl"):
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY saknas i miljövariablerna!")
    genai.configure(api_key=api_key)

    chunks = load_chunks(jsonl_path)
    contents = [chunk["content"] for chunk in chunks]
    embeddings = create_embeddings(contents)
    return chunks, embeddings


chunks, embeddings = initialize_rag()

st.title("The Ableton Live 12 MIDI Beginners Chatbot (RAG-Based)")

query = st.text_input("Ask your question here:")

if query:
    query_embedding = create_embeddings([query])[0]
    chunk_texts = [chunk["content"] for chunk in chunks]
    relevant_texts = semantic_search(query_embedding, chunk_texts, embeddings)
    context = "\n\n".join(relevant_texts)
    answer = generate_response(query, context)

    st.markdown("### Answer:")
    st.write(answer)
