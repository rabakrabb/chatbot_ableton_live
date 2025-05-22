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

/* Orange kant runt selectbox wrapper */
.stSelectbox > div > div {
    border: 2px solid var(--primaryColor) !important;
    border-radius: 6px !important;
    padding: 4px 8px !important;
    background-color: white !important;
}

/* Ta bort border på själva select och gör bakgrunden transparent */
.stSelectbox > div > div > select {
    border: none !important;
    background-color: transparent !important;
    color: black !important;
    width: 100% !important;
    font-family: var(--font-family) !important;
    padding: 6px 0 !important;
    outline: none !important;
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
    st.markdown("""
Test how well the chatbot performs by selecting a question from the list, seeing the AI's answer, then rating its quality using the scale below.
""")

    # Fördefinierade nybörjarfrågor med idealiska svar (facit)
    predefined_qa = [
        {
            "question": "What is a MIDI clip in Ableton Live?",
            "ideal_answer": "A MIDI clip is a block of MIDI notes and automation data that can be edited and played back in Ableton Live."
        },
        {
            "question": "How do I insert a virtual instrument in Ableton Live?",
            "ideal_answer": "You can insert a virtual instrument by dragging it from the Browser into a MIDI track."
        },
        {
            "question": "What does quantization do to MIDI notes?",
            "ideal_answer": "Quantization snaps MIDI notes to the nearest grid value to fix timing."
        },
        {
            "question": "How can I record MIDI input in Ableton Live?",
            "ideal_answer": "Arm the MIDI track and press the record button to capture MIDI input from your controller."
        },
        {
            "question": "What is the difference between a MIDI track and an audio track?",
            "ideal_answer": "A MIDI track contains MIDI data to trigger instruments, while an audio track contains recorded sound clips."
        },
    ]

    # Dropdown för att välja fråga
    st.markdown("### Select a predefined question:")
    question_idx = st.selectbox(
        "Choose a question to evaluate:",
        options=list(range(len(predefined_qa))),
        format_func=lambda x: predefined_qa[x]["question"],
        help="Select a question to test the chatbot on."
    )

    question = predefined_qa[question_idx]["question"]
    ideal_answer = predefined_qa[question_idx]["ideal_answer"]

    st.markdown(f"**Ideal answer:**  {ideal_answer}")

    # Generera chatbotens svar
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
    )

    if st.button("Submit Evaluation"):
        # Lägg till i session_state
        if "eval_scores" not in st.session_state:
            st.session_state.eval_scores = []
            st.session_state.eval_results = []

        st.session_state.eval_scores.append(rating)
        st.session_state.eval_results.append({
            "question": question,
            "ai_answer": model_answer,
            "ideal_answer": ideal_answer,
            "score": rating,
        })
        st.success("Evaluation saved!")

    # Visa historik
    if "eval_results" in st.session_state and st.session_state.eval_results:
        st.markdown("## Evaluation History")
        for i, res in enumerate(st.session_state.eval_results[::-1], 1):
            st.markdown(f"**Example {len(st.session_state.eval_results) - i + 1}**")
            st.markdown(f"- **Question:** {res['question']}")
            st.markdown(f"- **AI Answer:** {res['ai_answer']}")
            st.markdown(f"- **Ideal Answer:** {res['ideal_answer']}")
            st.markdown(f"- **Score:** {res['score']}")
            st.markdown("---")

        avg_score = sum(st.session_state.eval_scores) / len(st.session_state.eval_scores)
        st.markdown(f"### Session Average Score: `{avg_score:.2f}`")

    if st.button("Reset Evaluation Session"):
        st.session_state.eval_scores = []
        st.session_state.eval_results = []
