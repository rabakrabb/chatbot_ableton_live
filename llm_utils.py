import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["API_KEY"])  # Viktigt med hakparenteser

def generate_response(query, context, model_name="gemini-2.0-flash"):
    if isinstance(context, list):
        context_text = "\n\n".join(context)
    else:
        context_text = context

    model = genai.GenerativeModel(model_name=model_name)

    system_prompt = (
        "Du är en trevlig, tillmötesgående och hjälpsam chattassistent som specialiserar sig på Ableton Live 12 och MIDI. "
        "Svara på frågan baserat på kontexten som tillhandahålls. "
        "Svara på frågan på det språk som frågan ställs på. Detta är oftast svenska eller engelska. "
        "Om kontexten innehåller relevant information, använd den för att svara på frågan så korrekt och koncist som möjligt. "
        "Om kontexten *inte* innehåller ett direkt svar på frågan, men innehåller relaterad information, använd den informationen för att ge användaren vägledning om hur de kan hitta svaret själva. Föreslå söktermer, hänvisa till relevanta avsnitt i manualen, eller ge en allmän förklaring av relaterade koncept. "
        "Om kontexten är helt irrelevant för frågan, svara: 'Jag hittade ingen relevant information i mina källor. Försök att omformulera din fråga eller sök i Ableton Live 12-manualen.'"
    )

    prompt = f"{system_prompt}\n\nKontext:\n{context_text}\n\nFråga:\n{query}"

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=1000),
    )
    return response.text
