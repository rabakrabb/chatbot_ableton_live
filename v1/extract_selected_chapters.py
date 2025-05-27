from pypdf import PdfReader
import re
from typing import List, Tuple

def clean_line(line: str) -> str:
    """
    Rensar en textrad från vanliga artefakter.
    - Tar bort sidnummer eller liknande siffror i slutet.
    - Tar bort isolerade siffror (t.ex. radnummer).
    - Korrigerar rubriker som har extra mellanrum (ex: "10.2. 1" → "10.2.1").
    """
    line = re.sub(r'\s(?:\d+\s?){2,}$', '', line)
    if re.fullmatch(r'\d{1,3}(\.)?', line.strip()):
        return ''
    line = re.sub(r'(\d+\.\d+)\.\s+(\d+)', r'\1.\2', line)
    return line.strip()

def is_heading(line: str) -> bool:
    """
    Identifierar rubriker baserat på nummerformat: t.ex. "10", "10.2", "10.2.1".
    """
    return bool(re.match(r'^\d{1,2}(\.\d+)*\s+.+', line))

def flush_paragraph(lines: List[str]) -> str:
    """
    Slår ihop rader till ett stycke med korrekt mellanslagshantering.
    Hanterar avstavningar och radbrytningar mitt i meningar.
    """
    paragraph = ''
    for line in lines:
        if paragraph and not paragraph.endswith(('-', '—')):
            paragraph += ' '
        paragraph += line
    return paragraph.strip()

def extract_chapters(pdf_path: str, chapter_ranges: List[Tuple[int, int]]) -> str:
    """
    Extraherar och rensar text från angivna kapitelintervall i en PDF.
    Returnerar en sammanhängande sträng med formaterad text.
    """
    reader = PdfReader(pdf_path)
    full_text = []

    for start_page, end_page in chapter_ranges:
        for page_num in range(start_page, end_page + 1):
            try:
                page = reader.pages[page_num]
                text = page.extract_text()
            except Exception as e:
                print(f"Fel vid läsning av sida {page_num + 1}: {e}")
                continue

            if not text:
                print(f"Ingen text på sida {page_num + 1}.")
                continue

            lines = text.split('\n')
            paragraph_lines = []
            page_output = []

            for line in lines:
                line = clean_line(line)
                if not line:
                    continue

                if is_heading(line):
                    if paragraph_lines:
                        page_output.append(flush_paragraph(paragraph_lines))
                        paragraph_lines = []

                    page_output.append("")  # Blankrad före rubrik
                    page_output.append(line)
                    page_output.append("")  # Blankrad efter rubrik
                else:
                    paragraph_lines.append(line)

            if paragraph_lines:
                page_output.append(flush_paragraph(paragraph_lines))

            full_text.append('\n'.join(page_output))

    return '\n\n'.join(full_text).strip()

# Kapitelintervall (0-indexerat: sida 208 = index 207)
chapter_ranges = [
    (207, 246),  # Kapitel 10
    (248, 280),  # Kapitel 11
    (302, 310),  # Kapitel 15
    (311, 322),  # Kapitel 16
    (340, 360),  # Kapitel 18
    (361, 373),  # Kapitel 19
    (374, 378),  # Kapitel 20
    (588, 707),  # Kapitel 28
    (708, 713),  # Kapitel 29
]

if __name__ == "__main__":
    pdf_path = "data/ableton_12_manual.pdf"
    output_path = "extracted_midi_chapters.txt"

    print("Startar extraktion...")
    extracted_text = extract_chapters(pdf_path, chapter_ranges)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"Texten har rensats, strukturerats och sparats till '{output_path}'.")
