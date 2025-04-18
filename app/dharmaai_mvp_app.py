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
                completion = openai.ChatCompletion.create(
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

if streamlit_available:
    st.title("🪔 DharmaAI – Minimum Viable Conscience")
    st.subheader("Ask a question to GitaBot")

    st.markdown("""
        <style>
        .ask-another-button {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background-color: #ffe082;
            padding: 0.75rem 1.5rem;
            border-radius: 2rem;
            color: black;
            text-align: center;
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            cursor: pointer;
            transition: transform 0.2s ease, background-color 0.3s ease;
        }
        .ask-another-button:hover {
            background-color: #ffd54f;
            transform: scale(1.05);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='ask-another-button'>", unsafe_allow_html=True)
    if st.button("🔄 Ask Another"):
        st.session_state["user_input"] = ""
        if "Previous Questions" not in st.session_state:
            st.session_state["Previous Questions"] = []
        st.session_state["Previous Questions"].append("[Ask Another Clicked]")
        st.markdown("<script>window.scrollTo({ top: 0, behavior: 'smooth' });</script>", unsafe_allow_html=True)
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    mode = st.sidebar.radio("Select Mode", ["Krishna", "Krishna-GPT", "Krishna-Gemini", "Arjuna", "Vyasa", "Mirror", "Technical"])

    user_input = st.text_input("Your ethical question or dilemma:", value=st.session_state.get("user_input", ""), key="user_input")
    if st.button("🔍 Submit"):
        response = generate_gita_response(mode, df_matrix, user_input)
        st.markdown("""
        <div style='border: 1px solid #ddd; padding: 1rem; border-radius: 0.5rem; background-color: #f9f9f9; animation: fadeIn 0.8s ease-in;'>
        """, unsafe_allow_html=True)
        if 'verse_info' in locals() and verse_info is not None:
            st.markdown(f"<small>📘 Verse ID: {verse_info['Verse ID']} — <em>{verse_info['Symbolic Conscience Mapping']}</em></small>", unsafe_allow_html=True)
        st.markdown(response)
        st.markdown("</div>", unsafe_allow_html=True)

    if "Usage Journal" in st.session_state and st.session_state["Usage Journal"]:
        with st.expander("🕰️ View Past Interactions"):
            st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))
