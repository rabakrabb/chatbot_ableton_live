import re
from pypdf import PdfReader
from typing import List, Tuple

def clean_line(line: str) -> str:
    """
    Rensar en textrad från vanliga artefakter och formateringsproblem.
    - Tar bort sidnummer, isolerade siffror, och oönskade kortare strängar.
    - Korrigerar rubriker som har extra mellanslag (ex: "10.2. 1" → "10.2.1", "3.1 4" -> "3.14", "3 1.2" -> "31.2").
    """
    original_line = line.strip()
    cleaned_line = original_line

    # 1. Mycket aggressiv borttagning av sidnummer i slutet av en rad.
    cleaned_line = re.sub(r'\s+(?:\d+\s?){1,3}$', '', cleaned_line)

    # 2. Hantera helt tomma rader eller rader med bara mellanslag/bindestreck
    if not cleaned_line or re.fullmatch(r'^\s*$', cleaned_line) or re.fullmatch(r'^-+$', cleaned_line):
        return ''

    # 3. FÖRBÄTTRAD REGEL: Ta bort isolerade siffror/siffersekvenser som troligen är sidnummer eller artefakter.
    temp_normalized_line = re.sub(r'\s+', '', cleaned_line)
    temp_normalized_line = re.sub(r'\.+', '.', temp_normalized_line)
    if re.fullmatch(r'^\d+(\.\d+)*\.?$', temp_normalized_line):
        return ''

    # 4. Korrigera rubriker med mellanslag och felaktiga punkter. Ordern är viktig här!

    # NY REGEL: Kombinera initiala nummerdelar som "X Y" till "XY" om det följs av en punkt någonstans.
    # Ex: "1 7 . Routing" -> "17 . Routing"
    cleaned_line = re.sub(r'^(\d+)\s+(\d+)(\s*\.)', r'\1\2\3', cleaned_line)

    # Rule A: Hantera "X. Y" eller "X .Y" -> "X.Y" (t.ex. "10. 2" -> "10.2")
    # Tar bort mellanslag direkt intill en punkt i en nummerserie.
    # Ex: "16.7 .2" -> "16.7.2"
    cleaned_line = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', cleaned_line)

    # Rule C: Hantera "X.Y Z.A.B" -> "X.Y.Z.A.B" (t.ex. "26.1 1.1.2" -> "26.1.1.1.2")
    # Ersätter mellanslag med punkt om en siffra som innehåller en punkt följs av ett mellanslag
    # och sedan en annan siffra som också innehåller en punkt.
    cleaned_line = re.sub(r'(\d+\.\d+(?:\.\d+)*)\s+(\d+\.\d+(?:\.\d+)*)', r'\1.\2', cleaned_line)

    # Rule D: Hantera "X.Y Z" -> "X.YZ" (t.ex. "3.1 4" -> "3.14")
    # Konkatenerar om den andra delen är en ensam siffra som INTE följs av en punkt.
    cleaned_line = re.sub(r'(\d+\.\d+)\s+(\d)(?!\.)', r'\1\2', cleaned_line)
    
    # Rule E: Hantera "X.Y .Z" -> "X.Y.Z" (t.ex. "10.2 .1" -> "10.2.1")
    # Denna regel hanterar mellanslag som kan finnas mellan en punkt och nästa siffra i en serie.
    cleaned_line = re.sub(r'(\d+\.)\s+(\d+)', r'\1\2', cleaned_line)

    # NY REGEL: Ta bort mellanslag före punkt om det följs av icke-mellanslag (t.ex. text).
    # Ex: "17 . Routing" -> "17. Routing" (efter att "1 7" blivit "17")
    cleaned_line = re.sub(r'(\d+)\s*\.\s*(?=\S)', r'\1.', cleaned_line)

    # Rule F: Korrigera dubbla punkter som kan ha uppstått (t.ex. "10..1" -> "10.1", "10. .1" -> "10.1")
    cleaned_line = re.sub(r'\.\s*\.', '.', cleaned_line)
    cleaned_line = re.sub(r'\.\.+', '.', cleaned_line)

    # 5. Ta bort flera mellanslag, trimma
    cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()

    return cleaned_line


def is_heading(line: str) -> bool:
    """
    Identifierar rubriker baserat på nummerformat: t.ex. "10", "10.2", "10.2.1".
    Säkerställer att det är ett nummer följt av text.
    """
    # Regexet matchar:
    # ^\s* - Valfritt mellanslag i början
    # \d{1,2}        - 1-2 siffror (för huvudkapitel, t.ex. "10")
    # (?:\.\d+)* - Valfritt: en punkt följt av en eller flera siffror, noll eller flera gånger (för underkapitel, t.ex. ".2", ".1.4")
    # \.?            - Valfritt: en punkt efter det sista numret (t.ex. "2. Introduktion" vs "2 Introduktion")
    # \s+            - Ett eller flera mellanslag efter numret (måste finnas för att skilja från rena nummerartefakter)
    # .+             - En eller flera tecken (text som följer efter numret)
    # Använd raw string `r''` för regex för att undvika problem med backslashes.
    return bool(re.match(r'^\s*\d{1,2}(?:\.\d+)*\.?\s+.+', line))


def flush_paragraph(lines: List[str]) -> str:
    """
    Slår ihop rader till ett stycke med korrekt mellanslagshantering.
    Hanterar avstavningar och radbrytningar mitt i meningar genom att titta på föregående rad.
    """
    paragraph = []
    for i, line in enumerate(lines):
        # Om föregående rad (inte den första raden) slutade med ett bindestreck
        # (vanlig avstavning), då tar vi bort bindestrecket och lägger inte till mellanslag.
        if i > 0 and (paragraph[-1].endswith('-') or paragraph[-1].endswith('—')):
            paragraph[-1] = paragraph[-1].rstrip('—').rstrip('-')
            paragraph.append(line)
        else:
            # Lägg till mellanslag före den nya raden om det inte är början på stycket
            if paragraph:
                paragraph.append(' ')
            paragraph.append(line)

    # Sätt ihop allt och rensa upp eventuella dubbla mellanslag igen.
    # Använd raw string `r''` för regex för att undvika problem med backslashes.
    return re.sub(r'\s+', ' ', "".join(paragraph)).strip()


def extract_full_text_from_pdf(pdf_path: str) -> str:
    """
    Extraherar och rensar text från ALLA sidor i en PDF.
    Returnerar en sammanhängande sträng med formaterad text.
    """
    full_text: List[str] = []
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    print(f"Totala antalet sidor i PDF:en: {num_pages}")

    for i in range(num_pages):
        try:
            page = reader.pages[i]
            text = page.extract_text()
        except Exception as e:
            print(f"Fel vid läsning av sida {i + 1}: {e}")
            continue

        if not text:
            continue

        page_output: List[str] = []
        paragraph_lines: List[str] = []

        # Bearbeta rader individuellt
        for line in text.split('\n'):
            cleaned_line = clean_line(line) # Först rensa raden

            if not cleaned_line: # Hoppa över om raden blir tom efter rensning
                continue

            if is_heading(cleaned_line): # Kontrollera om det är en rubrik
                if paragraph_lines: # Flusha eventuella ackumulerade styckesrader
                    page_output.append(flush_paragraph(paragraph_lines))
                    paragraph_lines = []

                # Lägg till en tom rad före och efter rubriken för bättre läsbarhet och chunking
                page_output.append("")
                page_output.append(cleaned_line)
                page_output.append("")
            else:
                paragraph_lines.append(cleaned_line) # Annars är det en del av ett stycke

        # Flusha eventuella kvarvarande styckesrader från slutet av sidan
        if paragraph_lines:
            page_output.append(flush_paragraph(paragraph_lines))

        full_text.append('\n'.join(page_output))
        if (i + 1) % 50 == 0: # Utskrifter för att se framsteg var 50:e sida
            print(f"Bearbetat {i + 1}/{num_pages} sidor...")

    return '\n\n'.join(full_text).strip()

if __name__ == "__main__":
    pdf_file_path = r"C:\DS24\chatbot_ableton_live_full_manual\data\Ableton_12_manual.pdf" # Se till att denna fil finns
    output_file_path = "full_manual_text.txt" # Ny utfil för hela manualtexten

    print("Startar extraktion av hela manualen med förbättrad rensning och rubrikidentifiering...")
    extracted_text = extract_full_text_from_pdf(pdf_file_path)

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"Hela manualtexten har extraherats och sparats till '{output_file_path}'.")