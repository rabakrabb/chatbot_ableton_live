import streamlit as st
from typing import List, Dict
import json
import google.genai as genai


def create_embeddings(texts: List[str], model: str = "textembedding-gecko-001") -> List[List[float]]:
    """
    Skapar embeddings med Google GenAI.
    Kräver att API_KEY finns i .streamlit/secrets.toml.
    """
    api_key = st.secrets["API_KEY"]
    genai.configure(api_key=api_key)
    embedding_model = genai.GenerativeModel(model_name=model)

    embeddings = []
    for text in texts:
        response = embedding_model.embed_text(text)
        embeddings.append(response.embedding)

    return embeddings


def load_chunks(jsonl_path: str) -> List[Dict]:
    """
    Läser chunkade data från JSONL-fil.
    """
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks
