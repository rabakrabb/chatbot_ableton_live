import streamlit as st
from dotenv import load_dotenv
from vector_store import VectorStore
from llm_utils import generate_response
from rag_utils import create_embeddings, load_chunks

st.set_page_config(
    page_title="The Ableton Live 12 MIDI RAG-Bot",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🎹",
)

load_dotenv()

# --- CSS styling --- (Behåll som i din nuvarande app.py)
st.markdown("""
<style>
:root {
    --primaryColor: #FF6F61;
    --backgroundColor: #004D4D;
    --secondaryBackgroundColor: #006666;
    --font-family: "Arial, sans-serif";
    --textColor: white;
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

/* Fix för selectbox (dropdown) */
div[role="combobox"] > div > div > select {
    background-color: white !important;
    color: black !important;
    border: 2px solid var(--primaryColor) !important;
    border-radius: 6px !important;
    padding: 6px 8px !important;
    font-family: var(--font-family) !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def initialize_vector_store(jsonl_path: str = "chunks.jsonl") -> VectorStore:
    chunks = load_chunks(jsonl_path)
    chunks = [c for c in chunks if c.get("content", "").strip()]
    texts = [c["content"] for c in chunks]
    embeddings = create_embeddings(texts)

    store = VectorStore()
    for text, emb, meta in zip(texts, embeddings, chunks):
        store.add_item(text, emb, meta)
    return store

vector_store = initialize_vector_store()

# --- Meny ---
st.sidebar.title("Navigation")

st.sidebar.markdown("---")
answer_language = st.sidebar.selectbox(
    "Answer language:",
    options=["English", "Swedish"],
    index=0
)

page = st.sidebar.radio("Select a page", ["Chatbot", "Evaluation", "About the app"], index=0)

if page == "Chatbot":
    st.title("The Ableton Live 12 MIDI RAG-Bot")
    query = st.text_input("Ask your question:")
    if query:
        query_emb = create_embeddings([query])[0]
        results = vector_store.semantic_search(query_emb, k=5)
        top_texts = [r["text"] for r in results]
        answer = generate_response(query, "\n\n".join(top_texts), answer_language=answer_language)
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
    st.markdown("Test how well the chatbot performs by selecting a question and comparing the AI's answer with the ideal answer.")

    # ... Behåll din befintliga kod för predefined_qa_en och predefined_qa_sv ...

    # Välj rätt språklista
    predefined_qa = predefined_qa_en if answer_language == "English" else predefined_qa_sv

    st.markdown("### Select a predefined question:")
    question_idx = st.selectbox(
        "Choose a question:",
        options=list(range(len(predefined_qa))),
        format_func=lambda x: predefined_qa[x]["question"]
    )

    question = predefined_qa[question_idx]["question"]
    ideal_answer = predefined_qa[question_idx]["ideal_answer"]

    # Hämta svar från modellen via VectorStore
    query_emb = create_embeddings([question])[0]
    results = vector_store.semantic_search(query_emb, k=5)
    top_texts = [r["text"] for r in results]
    model_answer = generate_response(question, "\n\n".join(top_texts), answer_language=answer_language)

    st.markdown("### AI Assistant's answer:")
    st.write(model_answer)

    st.markdown("### Ideal answer:")
    st.write(ideal_answer)

    # Beräkna embedding-likhet
    model_emb = create_embeddings([model_answer])[0]
    ideal_emb = create_embeddings([ideal_answer])[0]

    from numpy import dot
    from numpy.linalg import norm

    similarity = dot(model_emb, ideal_emb) / (norm(model_emb) * norm(ideal_emb))
    score = round(similarity, 2)

    # Spara score i session state
    if "eval_scores" not in st.session_state:
        st.session_state.eval_scores = []

    st.session_state.eval_scores.append(score)

    st.markdown(f"### 🔍 Similarity Score: `{score}`")

    # Visa medelpoäng om minst 1 utvärdering
    if st.session_state.eval_scores:
        avg_score = sum(st.session_state.eval_scores) / len(st.session_state.eval_scores)
        st.markdown(f"### 🟢 Session Average Score: `{avg_score:.2f}`")

    if st.button("Reset Session Scores"):
        st.session_state.eval_scores = []
