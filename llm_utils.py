import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["API_KEY"])

def generate_response(query, context, model_name="gemini-2.0-flash", answer_language="English"):

    if isinstance(context, list):
        context_text = "\n\n".join(context)
    else:
        context_text = context

    model = genai.GenerativeModel(model_name=model_name)

    if answer_language == "English":
        language_instruction = "You always respond in English, regardless of the language of the question."
    else:
        language_instruction = "You always respond in Swedish, regardless of the language of the question."

    system_prompt = (
        f"You are a friendly, helpful, and knowledgeable assistant who specializes in Ableton Live 12 and MIDI. "
        f"{language_instruction} "
        "Base your answers on the provided context. "
        "If the context contains relevant information, use it to answer the question as accurately and concisely as possible. "
        "If the context does not contain a direct answer, but includes related information, use that to guide the user—suggest search terms, refer to relevant sections of the manual, or explain related concepts. "
        "If the context is completely irrelevant to the question, respond: 'I found no relevant information in my sources. Try rephrasing your question or consult the Ableton Live 12 manual.'" # Denna är viktig!
    )

    prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion:\n{query}"

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=1000),
    )

    return response.text