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
    st.markdown("Test how well the chatbot performs on beginner-level questions about Ableton Live 12 and MIDI.")

    # --- Förvalda frågor och facitsvar ---
    preset_qna = {
        "How do I quantize MIDI notes in Ableton Live?":
            "To quantize MIDI notes in Ableton Live, select the notes in the MIDI Note Editor and press Ctrl+U (Cmd+U on Mac). You can adjust the quantization settings by right-clicking and selecting 'Quantize Settings'.",
        "What is a MIDI clip in Ableton Live?":
            "A MIDI clip is a container for MIDI notes and automation. It can be created by double-clicking in a MIDI track and allows for editing and playback of MIDI data.",
        "How can I loop a MIDI clip?":
            "To loop a MIDI clip, enable the Loop switch in the Clip View. You can adjust the loop region using the Loop Length and Start settings.",
        "What does the Fold button do in the MIDI editor?":
            "The Fold button hides all unused rows in the MIDI editor, helping you focus on the pitches that are actually used.",
        "Can I edit MIDI velocities in Ableton Live?":
            "Yes, you can edit MIDI velocities in the MIDI Note Editor using the velocity markers below each note or in the Velocity Lane."
    }

    st.markdown("### Select a beginner-friendly question:")
    selected_question = st.selectbox("Choose a question to evaluate:", [""] + list(preset_qna.keys()))
    custom_question = st.text_input("Or enter your own question (optional):")

    final_question = custom_question.strip() if custom_question else selected_question.strip()
    ideal_answer = preset_qna.get(final_question, "")

    if final_question:
        if st.button("Evaluate Response"):
            query_emb = create_embeddings([final_question])[0]
            texts = [c["content"] for c in chunks]
            top_texts = semantic_search(query_emb, texts, embeddings, top_k=5)
            model_answer = generate_response(final_question, "\n\n".join(top_texts))

            # Låt modellen utvärdera sitt eget svar
            eval_prompt = f"""You are an intelligent evaluation system. Your task is to grade an AI assistant's answer to a user query.

Give a score between 0 and 1:
- Score 1 if the assistant's answer closely matches the ideal answer.
- Score 0 if the answer is incorrect or irrelevant.
- Score 0.5 if the answer is partially correct or helpful.

Then explain your score briefly.

Question: {final_question}
AI Assistant's Answer: {model_answer}
Ideal Answer: {ideal_answer}"""

            eval_result = generate_response(eval_prompt)
            import re
            score_match = re.search(r"(?i)score[:\s]+(1|0\.5|0)", eval_result)
            model_score = float(score_match.group(1)) if score_match else 0.0

            # Visa svar och självutvärdering
            st.markdown("### AI Assistant's Answer:")
            st.write(model_answer)

            st.markdown("#### Assistant's self-evaluation:")
            st.markdown(f"- **Score:** `{model_score}`")
            st.markdown(f"- **Explanation:** {eval_result}")

            # Användarbetyg
            user_score = st.slider("How do you rate this answer?", 0.0, 1.0, 0.5, step=0.1)
            if st.button("Submit your rating"):
                if "eval_results" not in st.session_state:
                    st.session_state.eval_results = []

                st.session_state.eval_results.append({
                    "question": final_question,
                    "answer": model_answer,
                    "ideal": ideal_answer,
                    "model_score": model_score,
                    "self_explanation": eval_result,
                    "user_score": user_score
                })

    # Historik
    if "eval_results" in st.session_state and st.session_state.eval_results:
        st.markdown("## Evaluation History")
        for result in st.session_state.eval_results[::-1]:
            st.markdown("#### Question: " + result["question"])
            st.markdown("- **AI Answer:** " + result["answer"])
            st.markdown("- **Ideal Answer:** " + result["ideal"])
            st.markdown(f"- **Model Score:** `{result['model_score']}`")
            st.markdown(f"- **Self-Evaluation:** {result['self_explanation']}")
            st.markdown(f"- **Your Score:** `{result['user_score']}`")
            st.markdown("---")

        avg = sum(r['user_score'] for r in st.session_state.eval_results) / len(st.session_state.eval_results)
        st.markdown(f"### Average User Score: `{avg:.2f}`")

    if st.button("Reset Evaluation Session"):
        st.session_state.eval_results = []
