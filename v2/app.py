import streamlit as st
from dotenv import load_dotenv
from vector_store import VectorStore
from llm_utils import generate_response
from rag_utils import create_embeddings # load_chunks behövs inte direkt i app.py längre
from numpy import dot
from numpy.linalg import norm
import os

st.set_page_config(
    page_title="The Ableton Live 12 RAG-Bot", # Uppdaterad titel
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

@st.cache_resource(show_spinner=False)
def initialize_vector_store() -> VectorStore:
    embeddings_parquet_path = "full_embeddings.parquet" # Sökväg till din nya embeddings-fil för hela manualen

    store = VectorStore()
    if store.load(embeddings_parquet_path): # Försök ladda från fil
        print(f"Vector store loaded from {embeddings_parquet_path}")
        return store
    else:
        st.error(f"Embeddingsfilen '{embeddings_parquet_path}' saknas. Vänligen kör 'generate_and_save_embeddings.py' först för att skapa den.")
        st.stop() # Stoppa appen om embeddings inte kan laddas

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
    st.title("The Ableton Live 12 RAG-Bot") # Uppdaterad titel
    query = st.text_input("Ask your question:")
    if query:
        query_emb = create_embeddings([query])[0]
        results = vector_store.semantic_search(query_emb, k=5)
        top_texts = [r["text"] for r in results]
        joined_texts = "\n\n".join([doc['content'] if isinstance(doc, dict) else doc.page_content for doc in top_texts])
        answer = generate_response(query, joined_texts, answer_language=answer_language)
        st.markdown("### Answer:")
        st.write(answer)

elif page == "About the app":
    st.title("About the app")
    st.write("""
This chatbot is built using semantic search and retrieval-augmented generation (RAG) to answer questions about Ableton Live 12.

It uses precomputed embeddings of the Ableton Live 12 manual chunks and Google's AI technology to generate context-aware answers.

Created by Martin Blomqvist during the Data Scientist program at EC Utbildning 2025.

For more information:
- [LinkedIn](https://www.linkedin.com/in/martin-blomqvist)
- [GitHub](https://github.com/rabakrabb)
""")

elif page == "Evaluation":
    st.title("Evaluate Chatbot Responses")
    st.markdown("Test how well the chatbot performs by selecting a question and comparing the AI's answer with the ideal answer.")

    # Frågor och svar på engelska (utökad för hela manualen)
    predefined_qa_en = [
        {"question": "How do I use automation in Ableton Live to change a parameter over time?", "ideal_answer": "Automation is drawn directly in tracks using breakpoint envelopes. Select the parameter you want to automate, and then draw its curve using the pen tool or by clicking and dragging breakpoints."},
        {"question": "What is the purpose of the Arrangement View in Ableton Live?", "ideal_answer": "The Arrangement View is a linear timeline for recording, arranging, and editing MIDI and audio clips in a traditional song structure."},
        {"question": "Explain the function of Sends and Returns in Ableton Live.", "ideal_answer": "Sends route a portion of a track's signal to a Return track, where effects can be applied. This allows multiple tracks to share the same effect processing, saving CPU and providing a consistent sound."},
        {"question": "How can I create a custom drum rack in Ableton Live?", "ideal_answer": "Drag individual samples or instruments into the pads of an empty Drum Rack. You can then configure each pad's settings and effects independently."},
        {"question": "What is the difference between hot-swapping and replacing a device?", "ideal_answer": "Hot-swapping allows you to audition different devices while keeping their current settings intact. Replacing a device permanently swaps it with a new one, discarding the old device's settings."},
        {"question": "How do I consolidate tracks or clips in Ableton Live?", "ideal_answer": "Select the desired clips or a range of time across multiple tracks, then go to the Edit menu and choose 'Consolidate Time to New Track' or 'Consolidate' (Cmd/Ctrl+J)."},
        {"question": "What are Scenes in the Session View and how are they used?", "ideal_answer": "Scenes in Session View are horizontal rows that contain a collection of clips, typically representing a section of a song. Launching a scene plays all clips within that row simultaneously, useful for live performance and improvisation."},
        {"question": "How can I reduce CPU usage in Ableton Live when my project is complex?", "ideal_answer": "To reduce CPU usage, you can freeze tracks, flatten tracks, reduce buffer size, disable unused devices, or use fewer CPU-intensive effects."},
        {"question": "Describe the function of the Follow Actions feature for MIDI and audio clips.", "ideal_answer": "Follow Actions allow you to define what happens after a clip finishes playing, such as playing another clip, stopping, retriggering itself, or launching a different scene. This is useful for creating dynamic arrangements and generative music."},
        {"question": "How do I set up an external MIDI controller in Ableton Live 12?", "ideal_answer": "Go to Live's Preferences, then 'Link/Tempo/MIDI'. Select your controller from the 'Control Surface' dropdown, enable its 'Track' and 'Remote' switches in the 'MIDI Ports' section, and ensure its MIDI input is active."}
    ]

    # Frågor och svar på svenska (utökad för hela manualen)
    predefined_qa_sv = [
        {"question": "Hur använder jag automation i Ableton Live för att ändra en parameter över tid?", "ideal_answer": "Automation ritas direkt i spåren med hjälp av brytpunktskuvert. Välj den parameter du vill automatisera och rita sedan dess kurva med pennverktyget eller genom att klicka och dra brytpunkter."},
        {"question": "Vad är syftet med Arrangement View i Ableton Live?", "ideal_answer": "Arrangement View är en linjär tidslinje för inspelning, arrangering och redigering av MIDI- och ljudklipp i en traditionell låtstruktur."},
        {"question": "Förklara funktionen av Sends och Returns i Ableton Live.", "ideal_answer": "Sends dirigerar en del av ett spårs signal till ett Return-spår, där effekter kan appliceras. Detta gör att flera spår kan dela samma effektprocessering, vilket sparar CPU och ger ett konsekvent ljud."},
        {"question": "Hur kan jag skapa ett anpassat Drum Rack i Ableton Live?", "ideal_answer": "Dra individuella samplingar eller instrument till padsen i ett tomt Drum Rack. Du kan sedan konfigurera varje pads inställningar och effekter oberoende av varandra."},
        {"question": "Vad är skillnaden mellan hot-swapping och att ersätta en enhet?", "ideal_answer": "Hot-swapping låter dig provlyssna olika enheter samtidigt som deras nuvarande inställningar behålls. Att ersätta en enhet byter ut den permanent mot en ny, vilket kasserar den gamla enhetens inställningar."},
        {"question": "Hur konsoliderar jag spår eller klipp i Ableton Live?", "ideal_answer": "Markera önskade klipp eller ett tidsintervall över flera spår, gå sedan till menyn Redigera och välj 'Consolidate Time to New Track' eller 'Consolidate' (Cmd/Ctrl+J)."},
        {"question": "Vad är Scener i Session View och hur används de?", "ideal_answer": "Scener i Session View är horisontella rader som innehåller en samling klipp, typiskt representerande en sektion av en låt. Att starta en scen spelar alla klipp inom den raden samtidigt, vilket är användbart för liveframträdanden och improvisation."},
        {"question": "Hur kan jag minska CPU-användningen i Ableton Live när mitt projekt är komplext?", "ideal_answer": "För att minska CPU-användningen kan du frysa spår, 'flattena' spår, minska buffertstorleken, inaktivera oanvända enheter eller använda färre CPU-intensiva effekter."},
        {"question": "Beskriv funktionen 'Follow Actions' för MIDI- och ljudklipp.", "ideal_answer": "Follow Actions låter dig definiera vad som händer efter att ett klipp spelats klart, till exempel att spela ett annat klipp, stoppa, återstarta sig själv, eller starta en annan scen. Detta är användbart för att skapa dynamiska arrangemang och generativ musik."},
        {"question": "Hur ställer jag in en extern MIDI-kontroller i Ableton Live 12?", "ideal_answer": "Gå till Lives inställningar, sedan 'Link/Tempo/MIDI'. Välj din kontroller från rullgardinsmenyn 'Control Surface', aktivera dess 'Track' och 'Remote' omkopplare i sektionen 'MIDI Ports', och se till att dess MIDI-ingång är aktiv."}
    ]

    # Välj språk
    predefined_qa = predefined_qa_en if answer_language == "English" else predefined_qa_sv

    st.markdown("### Select a predefined question:")
    question_idx = st.selectbox(
        "Choose a question:",
        options=list(range(len(predefined_qa))),
        format_func=lambda x: predefined_qa[x]["question"]
    )

    question = predefined_qa[question_idx]["question"]
    ideal_answer = predefined_qa[question_idx]["ideal_answer"]

    query_emb = create_embeddings([question])[0]
    results = vector_store.semantic_search(query_emb, k=15)
    top_texts = [r["text"] for r in results]
    joined_texts = "\n\n".join([doc['content'] if isinstance(doc, dict) else doc.page_content for doc in top_texts])
    model_answer = generate_response(question, joined_texts, answer_language=answer_language)


    st.markdown("### RAG-Bot's answer:")
    st.write(model_answer)

    st.markdown("### Ideal answer:")
    st.write(ideal_answer)

    no_answer_phrase = (
        "I found no relevant information in my sources. Try rephrasing your question or consult the Ableton Live 12 manual."
        if answer_language == "English"
        else "Jag hittade ingen relevant information i mina källor. Försök att omformulera din fråga eller konsultera Ableton Live 12 manualen."
    )

    if model_answer.strip() == no_answer_phrase.strip():
        score = 0.00
        st.markdown(f"### Similarity Score: `{score}` (AI did not provide an answer)")
    else:
        model_emb = create_embeddings([model_answer])[0]
        ideal_emb = create_embeddings([ideal_answer])[0]
        similarity = dot(model_emb, ideal_emb) / (norm(model_emb) * norm(ideal_emb))
        score = round(similarity, 2)
        st.markdown(f"### Similarity Score: `{score}`")

    if "eval_scores" not in st.session_state:
        st.session_state.eval_scores = []
    if "scored_ids" not in st.session_state:
        st.session_state.scored_ids = set()

    score_id = (question.strip(), model_answer.strip())
    if score_id not in st.session_state.scored_ids:
        st.session_state.eval_scores.append(score)
        st.session_state.scored_ids.add(score_id)

    valid_scores = [s for s in st.session_state.eval_scores if s is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        st.markdown(f"### Session Average Score: `{avg_score:.2f}` (based on {len(valid_scores)} evaluations)")

    if st.button("Reset Session Scores"):
        st.session_state.eval_scores = []
        st.session_state.scored_ids = set()
        st.rerun()

# --- Diskussion ---
# Min modell använder de specifika kapitel i Ableton Live 12-manualen som berör MIDI för att träna chatboten.
# Den är tränad för nybörjaren som vill lära sig om MIDI-musik och MIDI-skapande i programmet Ableton Live 12.
# I verkligheten kan denna chatbot användas av nya användare eller befintliga användare som vill ha snabba svar
# utan att behöva slå upp det i manualen. Det finns enligt mig stor potential i chatboten och den kan med enkelhet
# byggas vidare på för att hantera hela manualen och ge en mer övergripande hjälp.

# --- Potentiella utmaningar och möjligheter ---
# - Affärsmässiga
#     - Möjligheter:
#         - Kostnadsbesparingar, genom minskat tryck på kundsupport hos företaget.
#         - Skalbarhet, en bot kan hantera tusentals frågor samtidigt, en mänsklig support har svårt att tillgodose detta.
#         - 24/7 tillgänglighet, kundsupport dygnet runt, överallt.
#         - Konkurrensfördel, en välfungerande AI-support kan ge företaget fördelar gentemot konkurrenter.
#         - Engagemang/kundnöjdhet, att ha en effektiv bot nära till hands kan öka användarnöjdheten och lojaliteten.
#     - Utmaningar:
#         - Initiala kostnader, utveckling, träning och underhåll av en stabil och hållbar RAG-lösning kostar pengar och
#         kan innebära substantiella investeringar (API-kostnader för LLM, infrastruktur).
#         - Underhåll av datakällan, programmet uppdateras och manualen likaså, detta kräver ett kontinuerligt arbete för
#         att säkerställa kvalitet. Hur säkerställs att uppdateringar i manualen snabbt återspeglas i botens kunskapsbas?
#         Detta kräver en robust pipeline för dataflöde och uppdateringar av vektorbutiken.
#         - Kvalitetskontroll, felaktiga svar kan skada varumärket. Hur övervakas och kontrolleras botens svar lämpligt i
#         en större skala? Detta inkluderar att definiera metrics för svarskvalitet och att implementera feedbackmekanismer
#         från användare.
#         - Integration, hur kan boten integreras i befintlig supportstruktur eller inom Ableton Live-programvaran? Detta kan
#         vara tekniskt komplext och kräva anpassningar för att passa in i företagets befintliga ekosystem.
#         - Adoption, kommer användarna att ta till sig boten och lita på innehållet? Detta kan kräva tydliga evaluerings-
#         presentationer och en transparent kommunikation om botens kapacitet och begränsningar.

# - Etiska perspektiv:
#     - Hallucinationer och felaktigheter, även med en restriktiv systemprompt och säkerhetsåtgärder så kan en RAG-bot
#     "hallucinera" och hitta på fakta eller tolka kontexten felaktigt.
#         - Utmaningen ligger i hur dessa fall hanteras av användaren. Kan felaktiga instruktioner frustrera användare eller
#         till och med skada till exempel systeminställningar? Att agera på felaktiga instruktioner från en bot kan i värsta
#         fall leda till dataförlust eller att ett musikprojekt inte kan slutföras.
#         - Åtgärder kan här vara att till exempel vara tydlig med ansvarsfriskrivning, ge användaren möjlighet att rapportera
#         felaktigheter. En övervakning av konversationer kan identifiera problemområden. Likt i den aktuella Ableton-boten
#         bör systemprompten designas så att den uttryckligen endast svarar utifrån källmaterialet, med till exempel en så
#         kallad "no answer"-hantering. Detta sista steg är avgörande för att bygga förtroende och minimera felinformation.
#     - Bias, om den underliggande LLM:en har tränats på data som innehåller bias kan detta eventuellt "smitta av sig" på svaren,
#     även om en gedigen RAG-kontext minskar denna risk.
#         - Utmaningen kan vara huruvida manualen är respresentativ. Exempelvis kan vissa språkliga formuleringar, val av exempel,
#         eller betoning på vissa arbetsflöden oavsiktligt gynna specifika användargrupper eller musikgenrer, vilket kan leda till
#         att boten inte är lika hjälpsam för alla användare. Kan det finnas bias i språket eller strukturen - även om det
#         rör sig om en teknisk manual?
#         - Åtgärder för att hantera detta kan vara en kontinuerlig granskning av både den underliggande LLM:ens beteende och
#         källmaterialets innehåll. En transparent kommunikation om botens källor och begränsningar är också viktig.
#     - Användardata och sekretess, hur hanteras användardata där boten till exempel sparar konversationer för vidare utveckling
#     och utvärdering? Efterlever detta GDPR och liknande regelverk?
#         - Utmaningen, är användare införstådda och bekväma med att deras interaktioner loggas och övervakas? Bör detta vara
#         valfritt eller påverkar det kvaliteten på utvärderingen?
#         - Åtgärder, en tydlig integritetspolicy och möjlighet att avstå från datainsamling är avgörande för att bygga
#         användarförtroende och säkerställa en etisk drift.
#     - Transparens, är det tydligt och klart för användaren att den interagerar med en AI och inte en mänsklig supportagent?
#     Detta är viktigt för att hantera användarnas förväntningar. Om användare tror att de pratar med en människa kan det leda
#     till frustration när boten inte förstår mer komplexa problem eller brister i empati. En tydlig "Jag är en AI-assistent"-disclaimer
#     är en etisk standard.

# - Andra relevanta perspektiv:
#     - Tekniska begränsningar:
#         - Omfång, denna bot är begränsad till den data den är tränad på. Den kan inte svara på frågor utanför kontexten, vilket
#         kan vara frustrerande för användaren.
#         - Komplexa eller nyanserade frågor, LLM:er kan ha svårt att resonera kring och tyda väldigt komplexa flerstegs- eller
#         tvetydiga frågor, vilka kräver en djupare förståelse kring kontexten utanför det som står explicit i manualen.
#         - Latens/Svarstid, API-anrop till LLM:er kan ibland medföra latens, som kan påverka användarupplevelsen negativt om
#         svarstiden blir alltför lång. I en kreativ process kan en snabb respons vara avgörande för att upprätthålla användarens flow.
#         - Infrastruktur, att hosta och skala en RAG-bot i stor skala kan kräva en stor satsning på molninfrastruktur. vilket
#         ytterligare bidrar till driftskostnaderna.
#     - Användarupplevelse (UX):
#         - Den "no answer"-hantering, som denna bot inkluderar är viktigt utifrån att vara transparent med att boten inte
#         har något svar, och användaren kan tydligt se att den inte "gissar". Detta är en nyckelkomponent för att bibehålla användarens
#         förtroende.
#         - Interaktivitet, hur kan boten göras än mer interaktiv? I vilken mån ska den vidarebefordra användaren till en mänsklig
#         support, och när? Att implementera en sömlös överlämning till mänsklig agent för komplexa eller olösta frågor kan förbättra UX
#         avsevärt. Möjligheten att ställa följdfrågor, eller att boten proaktivt föreslår nästa steg eller relaterade ämnen baserat på
#         konversationen, kan också berika upplevelsen.

# --- Avslutning ---
# En RAG-bot av denna typ har enorm potential att revolutionera hur vi interagerar med information och support inom specifika områden
# och domäner. Med nödvändiga försiktighetsåtgärder, transparens och ett kontinuerligt kvalitetsarbete kan denna typ av lösningar bli
# ledande i framtidens informationsflöde. Genom att ta höjd för utmaningarna och sätta in effektiva motåtgärder kan nyttan med råge
# överväga riskerna. Ledordet här är att göra det kvalitativt, transparent, robust.