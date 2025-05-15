import streamlit as st
from typing import List, Dict
import json
import google.generativeai as genai

def create_embeddings(texts: List[str]) -> List[List[float]]:
    genai.configure(api_key=st.secrets["API_KEY"])
    model = genai.GenerativeModel("models/embedding-001") # Changed from EmbeddingModel to GenerativeModel
    embeddings = []
    for text in texts:
        response = model.embed_content(text) #changed from model.embed_content
        embeddings.append(response["embedding"]) # changed from response.embedding to response["embedding"]
    return embeddings


def load_chunks(jsonl_path: str) -> List[Dict]:
    chunks: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks
