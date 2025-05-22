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
    st.markdown("Test how well the chatbot performs by selecting a question and rating the AI's answer.")

    predefined_qa = [
        {"question": "What is a MIDI clip in Ableton Live?", "ideal_answer": "A MIDI clip is a block of MIDI notes and automation data that can be edited and played back in Ableton Live."},
        {"question": "How do I insert a virtual instrument in Ableton Live?", "ideal_answer": "Drag the instrument from the Browser into a MIDI track."},
        {"question": "What does quantization do to MIDI notes?", "ideal_answer": "Quantization snaps MIDI notes to the nearest grid value."},
        {"question": "How can I record MIDI input in Ableton Live?", "ideal_answer": "Arm a MIDI track and press record to capture input."},
        {"question": "What is the difference between a MIDI track and an audio track?", "ideal_answer": "MIDI tracks use note data to control instruments, while audio tracks play sound recordings."},
        {"question": "How do I create a new MIDI clip?", "ideal_answer": "Double-click an empty slot in a MIDI track to create a new clip."},
        {"question": "How do I open the piano roll?", "ideal_answer": "Double-click a MIDI clip to open the piano roll editor."},
        {"question": "How can I change note length in a MIDI clip?", "ideal_answer": "Select the note and drag its edge to adjust its duration."},
        {"question": "How do I loop a MIDI clip?", "ideal_answer": "Enable the loop button in the clip view."},
        {"question": "What is velocity in MIDI?", "ideal_answer": "Velocity controls how hard or soft a note is played."},
        {"question": "How can I duplicate a MIDI note?", "ideal_answer": "Select the note and press Ctrl+D (Cmd+D on Mac)."},
        {"question": "How do I delete a MIDI note?", "ideal_answer": "Select the note and press Delete or Backspace."}
    ]

    st.markdown("### Select a predefined question:")
    question_idx = st.selectbox(
        "Choose a question:",
        options=list(range(len(predefined_qa))),
        format_func=lambda x: predefined_qa[x]["question"]
    )

    question = predefined_qa[question_idx]["question"]
    ideal_answer = predefined_qa[question_idx]["ideal_answer"]
    st.markdown(f"**Ideal answer:**  {ideal_answer}")

    query_emb = create_embeddings([question])[0]
    texts = [c["content"] for c in chunks]
    top_texts = semantic_search(query_emb, texts, embeddings, top_k=5)
    model_answer = generate_response(question, "\n\n".join(top_texts))

    st.markdown("### AI Assistant's answer:")
    st.write(model_answer)

    st.markdown("### Rate the AI Assistant's answer:")
    rating = st.radio(
        "Choose a score",
        options=[0, 0.5, 1],
        format_func=lambda x: f"{x} {'(Bad)' if x == 0 else '(Partial)' if x == 0.5 else '(Good)'}",
        index=1,
        horizontal=True,
        key=f"rating_{question_idx}"
    )

    if "eval_scores" not in st.session_state:
        st.session_state.eval_scores = []

    if st.button("Submit rating"):
        st.session_state.eval_scores.append(rating)
        st.success("Thank you! Score submitted.")

    if st.session_state.get("eval_scores"):
        avg_score = sum(st.session_state.eval_scores) / len(st.session_state.eval_scores)
        st.markdown(f"### 🟢 Session Average Score: `{avg_score:.2f}`")

    if st.button("Reset Session Scores"):
        st.session_state.eval_scores = []
