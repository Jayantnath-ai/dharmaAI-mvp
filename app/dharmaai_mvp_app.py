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

if streamlit_available:
    st.set_page_config(page_title="DharmaAI MVP", layout="wide")

if openai_available:
    if not os.environ.get("OPENAI_API_KEY"):
        if streamlit_available:
            st.warning("⚠️ OpenAI API key not found. Krishna-GPT mode may not work.")
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    if streamlit_available:
        if "OPENAI_MODEL" not in st.session_state:
            st.session_state["OPENAI_MODEL"] = "gpt-3.5-turbo"
        available_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
        selected_model = st.sidebar.selectbox("🧠 Select OpenAI Model", available_models, index=available_models.index(st.session_state["OPENAI_MODEL"]))
        cost_per_1k = {
            "gpt-3.5-turbo": "$0.002 (input+output)",
            "gpt-4": "$0.05–$0.06 (est.)",
            "gpt-4o": "$0.02–$0.03 (est.)"
        }
        st.sidebar.caption(f"💰 Est. Cost per 1K tokens: {cost_per_1k.get(selected_model, 'Unknown')}")
        st.session_state["OPENAI_MODEL"] = selected_model

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Load verse matrix
matrix_paths = ["data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv", "app/data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv", "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"]
df_matrix = None
for path in matrix_paths:
    if os.path.exists(path):
        try:
            df_matrix = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df_matrix = pd.read_csv(path, encoding='ISO-8859-1')
        break

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

    if "Usage Journal" not in st.session_state:
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
        response = f"**🤖 Krishna-GPT says:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated GPT response here based on dharma logic]'}"
    elif mode == "Krishna-Gemini":
        response = f"**🌟 Krishna-Gemini reflects:**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated Gemini response to question]'}"
    elif mode == "Krishna":
        response = f"**🧠 Krishna teaches:**\n\n_You asked:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Symbolic dharma insight would be offered here]'}"
    elif mode == "Arjuna":
        response = f"**😟 Arjuna worries:**\n\n> What should I do about _'{user_input}'_?"
    elif mode == "Vyasa":
        response = f"**📖 Vyasa narrates:**\n\n> In the echoes of history, a seeker once asked: '{user_input}'"
    elif mode == "Mirror":
        response = "> You are not here to receive the answer.\n> You are here to see your reflection.\n> Ask again, and you may discover your dharma."
    elif mode == "Technical":
        response = f"🔧 Technical Mode:\nquestion: '{user_input}'\nrole_inferred: {user_role}\nmode_used: {mode}"

    if streamlit_available:
        st.session_state["Usage Journal"].append({
        "verse_id": verse_info['Verse ID'] if verse_info is not None else None,
        "mode": mode,
        "role": user_role,
        "question": user_input,
        "response": response,
        "tokens": total_tokens,
        "cost_usd": estimated_cost,
        "model": st.session_state.get("OPENAI_MODEL", "gpt-3.5-turbo")
            })

    return response
