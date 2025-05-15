import streamlit as st
import json
import google.generativeai as genai
from typing import List, Dict

def create_embeddings(texts: List[str], model: str = "textembedding-gecko-001") -> List[List[float]]:
    """
    Skapar embeddings med Google Generative AI.
    Kräver att API_KEY finns i Streamlit Secrets (secrets.toml eller via Streamlit Cloud).
    """
    api_key = st.secrets["API_KEY"]
    genai.configure(api_key=api_key)

    embeddings: List[List[float]] = []
    for text in texts:
        # Rätt sätt att hämta embedding i senaste SDK
        resp = genai.embeddings.get(model=model, text=text)
        embeddings.append(resp["embedding"])

    return embeddings

def load_chunks(jsonl_path: str) -> List[Dict]:
    """
    Läser in chunkade data från en JSONL-fil och returnerar en lista av dicts.
    Varje rad i JSONL ska vara ett objekt med fälten:
      - chunk_id
      - title
      - content
      - level
      - parent_chain
    """
    chunks: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks
