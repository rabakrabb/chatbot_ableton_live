import streamlit as st
from dotenv import load_dotenv
from vector_store import VectorStore
from llm_utils import generate_response
from rag_utils import create_embeddings, load_chunks
from numpy import dot
from numpy.linalg import norm

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
    "Response Language:",
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

    st.markdown("### RAG-Bot's answer:")
    st.write(model_answer)

    st.markdown("### Ideal answer:")
    st.write(ideal_answer)

    # Definiera "no-answer"-frasen baserat på valt språk
    if answer_language == "English":
        no_answer_phrase = "I found no relevant information in my sources. Try rephrasing your question or consult the Ableton Live 12 manual."
    else: # Swedish
        no_answer_phrase = "Jag hittade ingen relevant information i mina källor. Försök att omformulera din fråga eller konsultera Ableton Live 12 manualen."

    # --- Regelbaserad kontroll för "no-answer" ---
    if model_answer.strip() == no_answer_phrase.strip():
        score = 0.00 # Om AI:n explicit svarar med "no-answer"-frasen, sätt poängen till 0
        st.markdown(f"### Similarity Score: `{score}` (AI did not provide an answer)")
    else:
        # Beräkna embedding-likhet som tidigare
        model_emb = create_embeddings([model_answer])[0]
        ideal_emb = create_embeddings([ideal_answer])[0]

        similarity = dot(model_emb, ideal_emb) / (norm(model_emb) * norm(ideal_emb))
        score = round(similarity, 2)
        st.markdown(f"### Similarity Score: `{score}`")

    # Spara score i session state
    if "eval_scores" not in st.session_state:
        st.session_state.eval_scores = []

    # Se till att bara lägga till poängen en gång per fråga/svar-visning
    # Detta är en enkel mekanism, för en mer robust lösning kan du behöva lagra mer information
    # om den senast utvärderade frågan/svaret.
    if not st.session_state.eval_scores or st.session_state.eval_scores[-1] != score:
         st.session_state.eval_scores.append(score)


    # Visa medelpoäng om minst 1 utvärdering
    if st.session_state.eval_scores:
        avg_score = sum(st.session_state.eval_scores) / len(st.session_state.eval_scores)
        st.markdown(f"### Session Average Score: `{avg_score:.2f}`")

    if st.button("Reset Session Scores"):
        st.session_state.eval_scores = []

"""

--- Diskussion ---
Min modell använder de specifika kapitel i Ableton Live 12-manualen som berör MIDI för att träna chatboten. 
Den är tränad för nybörjaren som vill lära sig om MIDI-musik och MIDI-skapande i programmet Ableton Live 12.
I verkligheten kan denna chatbot användas av nya användare eller befintliga användare som vill ha snabba svar
utan att behöva slå upp det i manualen. Det finns enligt mig stor potential i chatboten och den kan med enkelhet
byggas vidare på för att hantera hela manualen och ge en mer övergripande hjälp.

--- Potentiella utmaningar och möjligheter ---
- Affärsmässiga
    - Möjligheter:
        - Konstnadsbesparingar, genom minskat tryck på kundsupport hos företaget.
        - Skalbarhet, en bot kan hantera tusentals frågor samtidigt, en mänsklig support har svårt att tillgodose detta.
        - 24/7 tillgänglighet, kundsupport dygnet runt, överallt.
        - Konkurrensfördel, en välfungerande AI-support kan ge företaget fördelar gentemot konkurrenter.
        - Engagemang/kundnöjdhet, att ha en effektiv bot nära till hands kan öka användarnöjdheten och lojaliteten.
    - Utmaningar:
        - Initiala kostnader, utveckling, träning och underhåll av en stabil och hållbar RAG-lösning kostar pengar och
        kan innebära substantiella investeringar (API-kostnader för LLM, infrastruktur).
        - Underhåll av datakällan, programmet uppdateras och manualen likaså, detta kräver ett kontinuerligt arbete för
        att säkerställa kvalitet. ur säkerställs att uppdateringar i manualen snabbt återspeglas i botens kunskapsbas? 
        Detta kräver en robust pipeline för dataflöde och uppdateringar av vektorbutiken.
        - Kvalitetskontroll, felaktiga svar kan skada varumärket. Hur övervakas och kontrolleras botens svar lämpligt i 
        en större skala? Detta inkluderar att definiera metrics för svarskvalitet och att implementera feedbackmekanismer 
        från användare.
        - Integration, hur kan boten integreras i befintlig supportstruktur eller inom Ableton Live-programvaran? Detta kan 
        vara tekniskt komplext och kräva anpassningar för att passa in i företagets befintliga ekosystem.
        - Adoption, kommer användarna att ta till sig boten och lita på innehållet? Detta kan kräva tydliga evaluerings-
        presentationer och en transparent kommunikation om botens kapacitet och begränsningar.

- Etiska perspektiv:
    - Hallucinationer och felaktigheter, även med en restriktiv systemprompt och säkerhetsåtgärder så kan en RAG-bot 
    "hallucinera" och hitta på fakta eller tolka kontexten felaktigt.
        - Utmaningen ligger i hur dessa fall hanteras av användaren. Kan felaktiga instruktioner frustrera användare eller
        till och med skada till exempel systeminställningar? Att agera på felaktiga instruktioner från en bot kan i värsta 
        fall leda till dataförlust eller att ett musikprojekt inte kan slutföras.
        - Åtgärder kan här vara att till exempel vara tydlig med ansvarsfriskrivning, ge användaren möjlighet att rapportera
        felaktigheter. En övervakning av konversationer kan identifiera problemområden. Likt i den aktuella Ableton-boten
        bör systemprompten designas så att den uttryckligen endast svarar utifrån källmaterialet, med till exempel en så
        kallad "no answer"-hantering. Detta sista steg är avgörande för att bygga förtroende och minimera felinformation.
    - Bias, om den underliggande LLM:en har tränats på data som innehåller bias kan detta eventuellt "smitta av sig" på svaren,
    även om en gedigen RAG-kontext minskar denna risk.
        - Utmaningen kan vara huruvida manualen är respresentativ. Exempelvis kan vissa språkliga formuleringar, val av exempel, 
        eller betoning på vissa arbetsflöden oavsiktligt gynna specifika användargrupper eller musikgenrer, vilket kan leda till 
        att boten inte är lika hjälpsam för alla användare. Kan det finnas bias i språket eller strukturen - även om det
        rör sig om en teknisk manual?
        - Åtgärder för att hantera detta kan vara en kontinuerlig granskning av både den underliggande LLM:ens beteende och 
        källmaterialets innehåll. En transparent kommunikation om botens källor och begränsningar är också viktig.
    - Användardata och sekretess, hur hanteras användardata där boten till exempel sparar konversationer för vidare utveckling
    och utvärdering? Efterlever detta GDPR och liknande regelverk?
        - Utmaningen, är användare införstådda och bekväma med att deras interaktioner loggas och övervakas? Bör detta vara 
        valfritt eller påverkar det kvaliteten på utvärderingen?
        - Åtgärder, en tydlig integritetspolicy och möjlighet att avstå från datainsamling är avgörande för att bygga 
        användarförtroende och säkerställa en etisk drift.
    - Transparens, är det tydligt och klart för användaren att den interagerar med en AI och inte en mänsklig supportagent?
    Detta är viktigt för att hantera användarnas förväntningar. Om användare tror att de pratar med en människa kan det leda 
    till frustration när boten inte förstår mer komplexa problem eller brister i empati. En tydlig "Jag är en AI-assistent"-disclaimer 
    är en etisk standard.

- Andra relevanta perspektiv:
    - Tekniska begränsningar:
        - Omfång, denna bot är begränsad till den data den är tränad på. Den kan inte svara på frågor utanför kontexten, vilket
        kan vara frustrerande för användaren.
        - Komplexa eller nyanserade frågor, LLM:er kan ha svårt att resonera kring och tyda väldigt komplexa flerstegs- eller
        tvetydiga frågor, vilka kräver en djupare förståelse kring kontexten utanför det som står explicit i manualen.
        - Latens/Svarstid, API-anrop till LLM:er kan ibland medföra latens, som kan påverka användarupplevelsen negativt om
        svarstiden blir alltför lång. I en kreativ process kan en snabb respons vara avgörande för att upprätthålla användarens flow.
        - Infrastruktur, att hosta och skala en RAG-bot i stor skala kan kräva en stor satsning på molninfrastruktur.
    - Användarupplevelse (UX):
        - Den "no answer"-hantering, som denna bot inkluderar är viktigt utifrån att vara transparent med att boten inte
        har något svar, och användaren kan tydligt se att den inte "gissar".
        - Interaktivitet, hur kan boten göras än mer interaktiv? I vilken mån ska den vidarebefordra användaren till en mänsklig
        support, och när?

--- Avslutning ---
En RAG-bot av denna typ har enorm potential att revolutionera hur vi interagerar med information och support inom specifika områden
och domäner. Med nödvändiga försiktighetsåtgärder, transparens och ett kontinuerligt kvalitetsarbete kan denna typ av lösningar bli
ledande i framtidens informationsflöde. Genom att ta höjd för utmaningarna och sätta in effektiva motåtgärder kan nyttan med råge
överväga riskerna. Ledordet här är att göra det kvalitativt, transparent, robust.

"""