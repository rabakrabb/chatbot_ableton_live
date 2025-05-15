import streamlit as st
import json
from rag_utils import create_embeddings, load_chunks
from llm_utils import generate_response
from semantic_search import semantic_search
from dotenv import load_dotenv
import google.generativeai as genai

# Ladda miljövariabler (behövs ej för st.secrets, men bra lokalt)
load_dotenv()

# Konfigurera API-nyckel för Google Generative AI via Streamlit Secrets
genai.configure(api_key=st.secrets["API_KEY"])

@st.cache_resource(show_spinner=False)
def initialize_rag(jsonl_path: str = "chunks.jsonl"):
    """
    Laddar chunkade dokument och skapar embeddings för dem.
    """
    # Läs in chunkade data från JSONL
    chunks = load_chunks(jsonl_path)
    # Extrahera bara textinnehållet
    contents = [chunk["content"] for chunk in chunks]
    # Skapa embeddings
    embeddings = create_embeddings(contents)
    return chunks, embeddings

# Initiera RAG (körs en gång och cachas)
chunks, embeddings = initialize_rag()

st.title("The Ableton Live 12 MIDI Beginners Chatbot (RAG-Based)")

query = st.text_input("Ask your question here:")

if query:
    # Skapa embedding för frågan
    query_embedding = create_embeddings([query])[0]
    # Semantisk sökning bland chunkade texter
    chunk_texts = [chunk["content"] for chunk in chunks]
    relevant_texts = semantic_search(query_embedding, chunk_texts, embeddings)
    # Sätt ihop kontext
    context = "\n\n".join(relevant_texts)
    # Generera svar
    answer = generate_response(query, context)

    st.markdown("### Answer:")
    st.write(answer)
