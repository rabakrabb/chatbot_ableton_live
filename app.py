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

    # Frågor och svar på engelska
    predefined_qa_en = [
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

    # Frågor och svar på svenska
    predefined_qa_sv = [
        {"question": "Vad är ett MIDI-klipp i Ableton Live?", "ideal_answer": "Ett MIDI-klipp är en sektion med MIDI-noter och automation som kan redigeras och spelas upp i Ableton Live."},
        {"question": "Hur lägger jag till ett virtuellt instrument i Ableton Live?", "ideal_answer": "Dra instrumentet från Browsern till ett MIDI-spår."},
        {"question": "Vad gör kvantisering med MIDI-noter?", "ideal_answer": "Kvantisering justerar MIDI-noter till närmaste rutnätsvärde."},
        {"question": "Hur kan jag spela in MIDI-inmatning i Ableton Live?", "ideal_answer": "Aktivera inspelning på ett MIDI-spår och tryck på record för att fånga inmatningen."},
        {"question": "Vad är skillnaden mellan ett MIDI-spår och ett ljudspår?", "ideal_answer": "MIDI-spår använder notdata för att styra instrument medan ljudspår spelar upp ljudinspelningar."},
        {"question": "Hur skapar jag ett nytt MIDI-klipp?", "ideal_answer": "Dubbelklicka på en tom plats i ett MIDI-spår för att skapa ett nytt klipp."},
        {"question": "Hur öppnar jag pianorullen?", "ideal_answer": "Dubbelklicka på ett MIDI-klipp för att öppna pianorullseditorn."},
        {"question": "Hur kan jag ändra notlängd i ett MIDI-klipp?", "ideal_answer": "Markera noten och dra i kanten för att justera dess längd."},
        {"question": "Hur loopar jag ett MIDI-klipp?", "ideal_answer": "Aktivera loop-knappen i klippvyn."},
        {"question": "Vad är velocity i MIDI?", "ideal_answer": "Velocity styr hur hårt eller mjukt en not spelas."},
        {"question": "Hur kan jag duplicera en MIDI-not?", "ideal_answer": "Markera noten och tryck Ctrl+D (Cmd+D på Mac)."},
        {"question": "Hur tar jag bort en MIDI-not?", "ideal_answer": "Markera noten och tryck Delete eller Backspace."}
    ]

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
