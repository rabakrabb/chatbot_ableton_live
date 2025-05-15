import streamlit as st
from typing import List, Dict
import json
import google.generativeai as genai

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Skapar embeddings för en lista av texter."""
    genai.configure(api_key=st.secrets["API_KEY"])

    embeddings = []
    for text in texts:
        try:
            response = genai.embed_content(model="models/embedding-001", content=text) # Här är ändringen
            embeddings.append(response['embedding'])
        except Exception as e:
            print(f"Error generating embedding for text: {text}")
            print(f"Error details: {e}")
            embeddings.append([])
    return embeddings

def load_chunks(jsonl_path: str) -> List[Dict]:
    """Läser in chunk-data från en JSONL-fil."""
    chunks = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File not found at {jsonl_path}")
        return []
    return chunks
