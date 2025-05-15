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
            response = genai.embed_content(
                model="models/embedding-001",
                content=text
            )
            embeddings.append(response['embedding'])
        except Exception as e:
            print(f"Error generating embedding for text: {text}")
            print(f"Error details: {e}")
            embeddings.append([])
    return embeddings
