Chatbot Ableton Live — Version 2
Det här är version 2 av den RAG-baserade chatboten för Ableton Live 12 manualen.

Funktioner
Extraherar och chunkar text från hela Ableton Live 12 manualen (PDF).
Skapar embeddings med Google GenAI:s embeddingmodell.
Semantisk sökning i chunkad data för att hitta relevant kontext.
Genererar svar med Google GenAI:s generativa modell baserat på kontext och fråga.
Streamlit-app med förbättrat gränssnitt och utökad funktionalitet.



Installation och användning


1. Klona repot och byt till v2-mappen:

git clone https://github.com/rabakrabb/chatbot_ableton_live.git
cd chatbot_ableton_live/v2



2. Skapa och aktivera virtuell miljö:

python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows


3. Installera dependencies:

pip install -r requirements.txt


4. Lägg till din Google GenAI API-nyckel i .streamlit/secrets.toml. Skapa mappen .streamlit och filen secrets.toml om de inte redan finns:

Ini, TOML

# .streamlit/secrets.toml
API_KEY="din_google_genai_api_nyckel_har"



Användning

Förbered manualen och generera chunks:

Placera din Ableton Live 12 PDF-manual i data/ableton_12_manual.pdf.


Kör skriptet för att extrahera text från PDF:en (detta skapar full_manual_text.txt):
python extract_selected_chapters.py


Kör därefter chunking-skriptet (detta skapar full_manual_chunks.jsonl):
python chunking.py


Generera och spara embeddings (detta skapar full_embeddings.parquet):
python generate_and_save_embeddings.py


Kör appen:

streamlit run app.py

Följ instruktionerna i webbläsaren för att ställa frågor.


Struktur

app.py: Streamlit frontend.

rag_utils.py: Funktioner för att skapa embeddings och ladda chunkad data.

llm_utils.py: Genererar svar med Google GenAI.

vector_store.py: Hanterar vector store för embeddings och utför semantisk sökning. (Observera att semantic_search.py är inkorporerad i vector_store.py i denna version, baserat på filerna.)

extract_selected_chapters.py: Skript för att extrahera text från PDF-manualen.

chunking.py: Chunkar extraherad text i lagom stora bitar.

generate_and_save_embeddings.py: Skript för att generera och spara embeddings från de chunkade filerna.

data/: Innehåller datafiler som den bearbetade manual-PDF:en (ableton_12_manual.pdf), extraherad text (full_manual_text.txt), chunkad data (full_manual_chunks.jsonl) och sparade embeddings (full_embeddings.parquet).
Gitignore


Lägg till följande i din .gitignore för att ignorera stora filer som manualen och genererade data:

# Datafiler
data/ableton_12_manual.pdf
data/full_manual_text.txt
data/full_manual_chunks.jsonl
data/full_embeddings.parquet

# Virtuell miljö
venv/
.venv/
Tips
Använd Git Large File Storage (LFS) för stora filer som manualen om du planerar att versionshantera den.
Spara API-nycklar i .streamlit/secrets.toml för säker hantering och undvik att committa dem direkt till ditt repo.
Uppdatera dependencies med pip freeze > requirements.txt efter att du har installerat nya paket.
