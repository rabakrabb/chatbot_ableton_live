import numpy as np
import polars as pl
import os # Lade till denna import

class VectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})


    def semantic_search(self, query_embedding, k=15):
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding)
        similarities = []
        for i, vector in enumerate(self.vectors):
            # Lägg till en kontroll för att undvika division med noll om normen är noll
            norm_query = np.linalg.norm(query_vector)
            norm_vector = np.linalg.norm(vector)
            if norm_query == 0 or norm_vector == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vector, vector) / (norm_query * norm_vector)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        return results

    def save(self, file_path: str = "embeddings.parquet"): # Nu korrekt indenterad
        df = pl.DataFrame(
            dict(
                vectors=self.vectors,
                texts=self.texts,
                metadata=self.metadata
            )
        )
        df.write_parquet(file_path)
        print(f"Vector store saved to {file_path}")

    def load(self, file_path: str = "embeddings.parquet"): # Nu korrekt indenterad
        if not os.path.exists(file_path):
            print(f"Error: Vector store file not found at {file_path}")
            return False
        df = pl.read_parquet(file_path)
        self.vectors = df["vectors"].to_list()
        self.texts = df["texts"].to_list()
        self.metadata = df["metadata"].to_list()
        # Konvertera listor till numpy arrayer igen om det behövs för att matcha add_item
        self.vectors = [np.array(vec) for vec in self.vectors]
        print(f"Vector store loaded from {file_path}")
        return True