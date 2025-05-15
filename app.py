import streamlit as st
import json
from typing import List
from semantic_search import semantic_search  # din befintliga
from llm_utils import generate_response    # din befintliga

st.set_page_config(page_title="Ableton Live 12 MIDI RAG-bot", layout="wide")

@st.cache_data
def load_chunks(jsonl_path: str = "chunks.jsonl"):
    """Läser in alla chunk-poster med färdiga embeddings."""
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

chunks = load_chunks()  # varje chunk har fälten: chunk_id, title, content, parent_chain, level, embedding

st.title("The Ableton Live 12 MIDI Beginners Chatbot (RAG-baserad)")
query = st.text_input("Ställ din fråga här:")

if query:
    # 1) Skapa embedding för frågan via färdigt GenAI‐anrop
    # (om du vill kan du också embedda frågan med genai.embeddings.get här,
    #  men för demo visar vi sem‐matchning på content‐fältet istället)
    query_embedding = chunks[0]["embedding"]  # dummy
    # 2) Semantisk sökning bland dina chunk-embeddings
    texts      = [c["content"]   for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    top_texts  = semantic_search(query_embedding, texts, embeddings, top_k=5)
    context    = "\n\n".join(top_texts)

    # 3) Generera svar från LLM givet frågan + kontexten
    answer = generate_response(query, context)

    st.markdown("### Svar:")
    st.write(answer)
