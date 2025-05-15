import streamlit as st
from typing import List, Dict
import json
import google.generativeai as genai

def create_embeddings(texts: List[str]) -> List[List[float]]:
    genai.configure(api_key=st.secrets["API_KEY"])

    embedding_model = genai.embedder.EmbeddingModel(model_name="models/embedding-001")

    embeddings = []
    for text in texts:
        response = embedding_model.embed(content=text)
        embeddings.append(response['embedding'])

    return embeddings


def load_chunks(jsonl_path: str) -> List[Dict]:
    chunks: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks
