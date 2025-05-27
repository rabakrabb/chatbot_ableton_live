import re
import json
from typing import List, Dict, Optional


class Chunk:
    def __init__(
        self,
        chunk_id: str,
        title: str,
        content: str,
        level: str,
        parent_chain: List[Dict[str, str]],
    ):
        self.chunk_id = chunk_id
        self.title = title
        self.content = content
        self.level = level
        self.parent_chain = parent_chain  # [{"chunk_id":..., "title":...}, ...]

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "parent_chain": self.parent_chain,
        }


def determine_level(chunk_id: str) -> str:
    """Bestämmer nivån på chunk baserat på antal punkter i chunk_id."""
    count = chunk_id.count(".")
    if count == 0:
        return "main"
    elif count == 1:
        return "sub"
    elif count == 2:
        return "subsub"
    else:
        return "deep"


def update_parent_chain(
    current_id: str, current_title: str, parent_chain: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Uppdaterar föräldrakedjan för aktuell chunk.
    Behåller bara prefix-prefix som matchar och lägger till aktuell chunk.
    """
    new_chain = []
    parts = current_id.split(".")

    for i in range(len(parts) - 1):
        parent_id = ".".join(parts[: i + 1])
        parent_title = ""
        # Försök hitta titeln i den befintliga parent_chain eller antag en standardtitel
        for p in parent_chain:
            if p["chunk_id"] == parent_id:
                parent_title = p["title"]
                break
        if not parent_title: # Om ingen titel hittades, använd en generisk
            parent_title = f"Kapitel {parent_id}" # Kan justeras vid behov
        new_chain.append({"chunk_id": parent_id, "title": parent_title})

    # Lägg till aktuell chunk i kedjan
    new_chain.append({"chunk_id": current_id, "title": current_title})

    return new_chain


def chunk_text_from_file(input_path: str, output_path: str):
    """
    Läser textfil och chunkar enligt numrerade rubriker.
    Sparar chunks som JSONL med metadata.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks: List[Chunk] = []
    current_chunk_id: Optional[str] = None
    current_title: str = ""
    current_content: List[str] = []
    parent_chain: List[Dict[str, str]] = [] # Denna håller den *aktuella* föräldrakedjan för nästa chunk

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Matcha huvudrubrik (t.ex. "10." eller "10 Kapitelnamn")
        match_main = re.match(r"^(\d+)\.\s*(.*)$", line)
        if match_main:
            if current_chunk_id is not None:
                chunks.append(
                    Chunk(
                        chunk_id=current_chunk_id,
                        title=current_title,
                        content=" ".join(current_content).strip(),
                        level=determine_level(current_chunk_id),
                        parent_chain=parent_chain[:-1].copy(), # Exkludera den nuvarande chunken från sin egen parent_chain
                    )
                )
            current_chunk_id = match_main.group(1)
            current_title = match_main.group(2) or f"Kapitel {current_chunk_id}"
            current_content = []
            parent_chain = [{"chunk_id": current_chunk_id, "title": current_title}] # Nollställ föräldrakedjan för en ny huvudrubrik
            continue

        # Matcha underrubrik (t.ex. "10.2.1 Editing Notes")
        match_sub = re.match(r"^(\d+(?:\.\d+)+)\s+(.+)", line)
        if match_sub:
            if current_chunk_id is not None:
                chunks.append(
                    Chunk(
                        chunk_id=current_chunk_id,
                        title=current_title,
                        content=" ".join(current_content).strip(),
                        level=determine_level(current_chunk_id),
                        parent_chain=parent_chain[:-1].copy(), # Exkludera den nuvarande chunken från sin egen parent_chain
                    )
                )
            current_chunk_id = match_sub.group(1)
            current_title = match_sub.group(2)
            # Uppdatera parent_chain med den nya underrubriken som sista element
            parent_chain = update_parent_chain(current_chunk_id, current_title, parent_chain)
            current_content = []
            continue

        # Annars är det vanlig text
        current_content.append(line)

    # Spara sista chunk efter loopen
    if current_chunk_id is not None:
        chunks.append(
            Chunk(
                chunk_id=current_chunk_id,
                title=current_title,
                content=" ".join(current_content).strip(),
                level=determine_level(current_chunk_id),
                parent_chain=parent_chain[:-1].copy(), # Exkludera den nuvarande chunken från sin egen parent_chain
            )
        )

    # Skriv till JSONL
    with open(output_path, "w", encoding="utf-8") as out_file:
        for chunk in chunks:
            json.dump(chunk.to_dict(), out_file, ensure_ascii=False)
            out_file.write("\n")

    print(f"Chunkning klar! {len(chunks)} chunks sparade i '{output_path}'.")
    

if __name__ == "__main__":
    # Observera användningen av r"" för att hantera backslashes korrekt i Windows-vägar.
    # Du måste köra denna kod på din egen dator för att den ska kunna läsa filen.
    input_file = r"C:\DS24\chatbot_ableton_live_full_manual\data\full_manual_text.txt"
    output_file = "full_manual_chunks.jsonl" # Ny utfil för chunks

    # Korrigerat funktionsanrop från chunk_document till chunk_text_from_file
    chunk_text_from_file(input_file, output_file)