# Chatbot Ableton Live MIDI

Det här projektet är en RAG-baserad chatbot för att söka och ställa frågor om Ableton Live 12 manualen, med fokus på MIDI-relaterade kapitel.

## Funktioner

- Extraherar och chunkar relevant text från Ableton Live 12 manual (PDF).
- Skapar embeddings med Google GenAI:s embeddingmodell.
- Semantic search i chunkad data för att hitta relevant kontext.
- Genererar svar med Google GenAI:s generativa modell baserat på kontext och fråga.
- Streamlit-app för användarvänlig chattgränssnitt.

## Installation

1. Klona repo:

git clone https://github.com/rabakrabb/chatbot_ableton_live_midi.git
cd chatbot_ableton_live_midi


2. Skapa och aktivera virtuell miljö:

python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows


3. Installera dependencies:

pip install -r requirements.txt


4. Lägg till din Google GenAI API-nyckel i .streamlit/secrets.toml:

API_KEY="din_google_genai_api_nyckel_har"


Användning
Kör appen med:

streamlit run app.py

Följ instruktionerna i webbläsaren för att ställa frågor.


Struktur
app.py: Streamlit frontend.

rag_utils.py: Funktioner för embedding och laddning av chunkad data.

llm_utils.py: Genererar svar med Google GenAI.

semantic_search.py: Semantic search-funktioner.

extract_selected_chapters.py: Skript för att extrahera text från PDF.

chunking.py: Chunkar extraherad text i lagom stora bitar.

data/: Datafiler och manual PDF.


Gitignore
Lägg till i .gitignore för att ignorera stora filer som manualen:

data/ableton_12_manual.pdf


Tips
Använd Git Large File Storage (LFS) för stora filer som manualen.

Spara API-nycklar i secrets.toml för säker hantering.

Uppdatera dependencies med pip freeze > requirements.txt efter nya paket.