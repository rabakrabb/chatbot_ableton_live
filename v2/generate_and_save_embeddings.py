# generate_and_save_embeddings.py
from dotenv import load_dotenv
from vector_store import VectorStore
from rag_utils import create_embeddings, load_chunks
import os
import time

load_dotenv() # Ladda API-nycklar

def main():
    jsonl_path = r"data\full_manual_chunks.jsonl" # Din nya chunk-fil
    output_parquet_path = r"data\full_embeddings.parquet" # Fil där embeddings sparas

    if os.path.exists(output_parquet_path):
        print(f"Embeddingsfilen '{output_parquet_path}' finns redan. Hoppar över generering.")
        print("Om du vill generera om, radera filen först.")
        return

    chunks = load_chunks(jsonl_path)
    chunks = [c for c in chunks if c.get("content", "").strip()]

    if not chunks:
        print("Inga chunks hittades. Se till att 'full_manual_chunks.jsonl' är korrekt.")
        return

    texts = [c["content"] for c in chunks]

    print(f"Genererar embeddings för {len(texts)} chunks. Detta kan ta lång tid och kosta pengar...")
    start_time = time.time()

    # Generera embeddings i batchar för att undvika överbelastning av API:et och för bättre hantering
    batch_size = 100 # Justera detta baserat på API-limiteringar och minne
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = create_embeddings(batch_texts)
        all_embeddings.extend(batch_embeddings)
        print(f"Genererat embeddings för {min(i + batch_size, len(texts))}/{len(texts)} chunks. Tid: {time.time() - start_time:.2f} sekunder.")
        time.sleep(1) # Paus för att respektera API-rate limits

    if len(all_embeddings) != len(texts):
        print("Varning: Antalet genererade embeddings matchar inte antalet texter.")
        # Hantera felaktiga embeddings här om de tillåts (t.ex. tomma listor)
        # Filter out empty embeddings if create_embeddings returns them on error
        filtered_embeddings = [emb for emb in all_embeddings if emb]
        filtered_texts = [texts[i] for i, emb in enumerate(all_embeddings) if emb]
        filtered_chunks = [chunks[i] for i, emb in enumerate(all_embeddings) if emb]

        if len(filtered_embeddings) != len(texts):
            print(f"Fortsätter med {len(filtered_embeddings)} giltiga embeddings.")
            all_embeddings = filtered_embeddings
            texts = filtered_texts
            chunks = filtered_chunks


    store = VectorStore()
    for text, emb, meta in zip(texts, all_embeddings, chunks):
        store.add_item(text, emb, meta)

    store.save(output_parquet_path) # Din save-metod behöver nog en sökväg som parameter
    print(f"Embeddings sparade till '{output_parquet_path}'. Total tid: {time.time() - start_time:.2f} sekunder.")

if __name__ == "__main__":
    main()