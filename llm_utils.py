import google.genai as genai
import streamlit as st
from typing import Union, List


genai.configure(api_key=st.secrets["API_KEY"])


def generate_response(
    query: str, context: Union[str, List[str]], model_name: str = "gemini-2.0-flash"
) -> str:
    """
    Genererar svar från Google GenAI.
    Kontext kan vara sträng eller lista av strängar (slås ihop).
    """
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
        generation_config=genai.types.GenerationConfig(max_output_tokens=1000)
    )
    return response.text
