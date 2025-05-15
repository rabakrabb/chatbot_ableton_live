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
        "Du är en hjälpsam chattassistent. "
        "Svara på frågan baserat på kontexten som tillhandahålls. "
        "Om kontexten inte innehåller svaret, svara: 'Jag hittade inget relevant svar i databasen.'"
    )

    prompt = f"{system_prompt}\n\nKontext:\n{context_text}\n\nFråga:\n{query}"

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=1000),
    )
    return response.text
