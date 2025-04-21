import sys
import random
import os
import re

try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

try:
    import streamlit as st
    streamlit_available = True
except ImportError:
    streamlit_available = False

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import google.generativeai as genai
    gemini_available = True
except ImportError:
    gemini_available = False

# Define generate_gita_response inline
def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or len(user_input.strip()) < 5:
        return "🛑 Please ask a more complete or meaningful question."

    import numpy as np

    def get_embedding(text):
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    user_role = "seeker"
    token_multiplier = 1.25
    prompt_tokens = int(len(user_input.split()) * token_multiplier)
    response_tokens = 120
    total_tokens = prompt_tokens + response_tokens
    estimated_cost = round((total_tokens / 1000) * 0.002, 6)

    if streamlit_available and "Usage Journal" not in st.session_state:
        st.session_state["Usage Journal"] = []

    response = ""
    verse_info = None
    if df_matrix is not None and not df_matrix.empty:
        if 'embedding' not in df_matrix.columns:
            df_matrix['embedding'] = df_matrix['Short English Translation'].apply(lambda x: get_embedding(str(x)))
        user_embedding = get_embedding(user_input)
        df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))
        verse_info = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]

    if mode == "Krishna-GPT":
        if openai_available and os.getenv("OPENAI_API_KEY"):
            try:
                system_prompt = f"You are Krishna from the Bhagavad Gita. Provide dharma-aligned, symbolic, and contextual guidance. Verse context: '{verse_info['Short English Translation']}' with symbolic tag '{verse_info['Symbolic Conscience Mapping']}'"
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                completion = client.chat.completions.create(
                    model=st.session_state.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7
                )
                reply = completion.choices[0].message.content.strip()
                response = f"**🤖 Krishna-GPT says:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {reply}"
            except Exception as e:
                response = f"❌ Error fetching response from Krishna-GPT: {str(e)}"
        else:
            response = f"**🤖 Krishna-GPT says:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated GPT response here based on dharma logic]'}"

    elif mode == "Krishna-Gemini":
        if gemini_available:
            try:
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if gemini_api_key:
                    genai.configure(api_key=gemini_api_key)
                    available_models = genai.list_models()
                    selected_gemini_model = st.sidebar.selectbox("Choose Gemini model", ["models/gemini-1.5-flash", "models/gemini-1.5-pro"], index=0)
                    model_name = selected_gemini_model
                    model = genai.GenerativeModel(model_name)
                    gemini_prompt = f"You are Krishna. Reflect and answer this question with dharmic insight: '{user_input}'. Gita Verse: '{verse_info['Short English Translation']}' tagged '{verse_info['Symbolic Conscience Mapping']}'"
                    gemini_reply = model.generate_content(gemini_prompt)
                    response = f"**🌟 Krishna-Gemini reflects:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {gemini_reply.text.strip()}"
                else:
                    response = f"⚠️ Gemini API key not set.\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated Gemini response]'}"
            except Exception as e:
                response = f"❌ Error fetching response from Krishna-Gemini: {str(e)}"
        else:
            response = "⚠️ Gemini module not available."

    return response
