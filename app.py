import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv

# Egna moduler
from semantic_search import semantic_search
from llm_utils import generate_response
from rag_utils import create_embeddings, load_chunks

st.set_page_config(
    page_title="The Ableton Live 12 MIDI RAG-Bot",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🎹",
)

load_dotenv()

# --- CSS styling ---
st.markdown("""
<style>
:root {
    --primaryColor: #FF6F61;
    --backgroundColor: #004D4D;
    --secondaryBackgroundColor: #006666;
    --font-family: "Arial, sans-serif";
}
body, main {
    margin: 0; padding: 0; min-height: 100vh;
    background-color: var(--backgroundColor) !important;
    color: var(--textColor) !important;
    font-family: var(--font-family) !important;
}
section.main, div.block-container {
    max-width: 800px !important;
    margin: 80px auto 40px auto !important;
    background-color: var(--secondaryBackgroundColor) !important;
    padding: 30px 40px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    color: var(--textColor) !important;
}
h1, h2, h3 {
    color: var(--primaryColor) !important;
    font-weight: 700 !important;
    margin-top: 0 !important;
}
.stTextInput > div > div > input {
    width: 100% !important;
    background-color: white !important;
    color: black !important;
    border: 2px solid var(--primaryColor) !important;
    border-radius: 6px !important;
    padding: 8px !important;
    font-family: var(--font-family) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #888 !important;
}
div.stButton > button {
    background-color: var(--primaryColor) !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease !important;
    font-family: var(--font-family) !important;
}
div.stButton > button:hover {
    background-color: #e65b50 !important;
}
[data-testid="stSidebar"] {
    background-color: var(--secondaryBackgroundColor) !important;
}
[data-testid="stSidebar"] * {
    color: var(--textColor) !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def initialize_rag(jsonl_path: str = "chunks.jsonl"):
    chunks = load_chunks(jsonl_path)
    chunks = [c for c in chunks if c.get("content", "").strip()]
    embeddings = create_embeddings([c["content"] for c in chunks])
    return chunks, embeddings

chunks, embeddings = initialize_rag()

# --- Meny ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Chatbot", "Evaluation", "About the app"], index=0)

if page == "Chatbot":
    st.title("The Ableton Live 12 MIDI RAG-Bot")
    query = st.text_input("Ask your question:")  # ✅ Label tydligt
    if query:
        query_emb = create_embeddings([query])[0]
        texts = [c["content"] for c in chunks]
        top_texts = semantic_search(query_emb, texts, embeddings, top_k=5)
        answer = generate_response(query, "\n\n".join(top_texts))
        st.markdown("### Answer:")
        st.write(answer)

elif page == "About the app":
    st.title("About the app")
    st.write("""
This chatbot is built using semantic search and retrieval-augmented generation (RAG) to answer questions about Ableton Live 12 MIDI fuctions and its virtual instruments.

It uses precomputed embeddings of the Ableton Live 12 manual chunks and Google's AI technology to generate context-aware answers.

Created by Martin Blomqvist during the Data Scientist program at EC Utbildning 2025.

For more information:
- [LinkedIn](https://www.linkedin.com/in/martin-blomqvist)
- [GitHub](https://github.com/rabakrabb)
""")

elif page == "Evaluation":
    st.title("Evaluate Chatbot Responses")
    st.markdown("Test how well the chatbot performs by submitting multiple questions and comparing its answers to ideal responses.")

    if "eval_scores" not in st.session_state:
        st.session_state.eval_scores = []
        st.session_state.eval_results = []

    # ✅ Alla inputs har nu labels
    question = st.text_input("User question", key="eval_q")
    ideal_answer = st.text_area("Ideal answer", key="eval_a")
    
    if st.button("Evaluate response"):
        if question and ideal_answer:
            query_emb = create_embeddings([question])[0]
            texts = [c["content"] for c in chunks]
            top_texts = semantic_search(query_emb, texts, embeddings, top_k=5)
            model_answer = generate_response(question, "\n\n".join(top_texts))

            evaluation_system_prompt = """You are an intelligent evaluation system. Your task is to grade an AI assistant's answer to a user query.

Give a score between 0 and 1:
- Score 1 if the assistant's answer closely matches the ideal answer.
- Score 0 if the answer is incorrect or irrelevant.
- Score 0.5 if the answer is partially correct or helpful.

Then explain your score briefly.
"""
            evaluation_prompt = f"""Question: {question}
AI Assistant's Answer: {model_answer}
Ideal Answer: {ideal_answer}"""

            eval_result = generate_response(evaluation_system_prompt, evaluation_prompt)

            import re
            score_match = re.search(r"(?i)score[:\s]+(1|0\.5|0)", eval_result)
            score = float(score_match.group(1)) if score_match else 0.0

            st.session_state.eval_scores.append(score)
            st.session_state.eval_results.append({
                "question": question,
                "answer": model_answer,
                "ideal": ideal_answer,
                "score": score,
                "explanation": eval_result
            })

    if st.session_state.eval_results:
        st.markdown("## Evaluation History")
        for i, result in enumerate(st.session_state.eval_results[::-1], 1):
            st.markdown(f"**Example {len(st.session_state.eval_results) - i + 1}**")
            st.markdown(f"- **Question:** {result['question']}")
            st.markdown(f"- **AI Answer:** {result['answer']}")
            st.markdown(f"- **Ideal Answer:** {result['ideal']}")
            st.markdown(f"- **Score:** {result['score']}")
            st.markdown(f"- **Explanation:** {result['explanation']}")
            st.markdown("---")

        avg_score = sum(st.session_state.eval_scores) / len(st.session_state.eval_scores)
        st.markdown(f"### Session Average Score: `{avg_score:.2f}`")

    if st.button("Reset Evaluation Session"):
        st.session_state.eval_scores = []
        st.session_state.eval_results = []
