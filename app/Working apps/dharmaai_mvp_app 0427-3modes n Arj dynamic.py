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

    if mode == "Krishna-Explains":
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
                response = f"**🤖 Krishna-Explains :**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {reply}"
            except Exception as e:
                response = f"❌ Error fetching response from Krishna-Explains: {str(e)}"
        else:
            response = f"**🤖 Krishna-Explains says:**\n\n_Reflecting on your question:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated GPT response here based on dharma logic]'}"

    elif mode == "Krishna":
        response = f"**🧠 Krishna teaches:**\n\n_You asked:_ **{user_input}**\n\n> {verse_info['Short English Translation'] if verse_info is not None else '[Symbolic dharma insight would be offered here]'}"

    # 🌀 New Dynamic Arjuna Mode
    elif mode == "Arjuna":
        reflections = generate_arjuna_reflections(user_input)
        response = (
            f"**😟 Arjuna's Reflections:**\n\n"
            f"_Reflecting on your question:_ **'{user_input}'**\n\n"
            f"Here are three doubts arising in my mind:\n\n"
            f"1. {reflections[0]}\n"
            f"2. {reflections[1]}\n"
            f"3. {reflections[2]}"
        )

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

# 🧠 Dynamic Arjuna Reflection Helper
def generate_arjuna_reflections(user_input):
    import random

    themes = [
        "fear", "attachment", "purpose", "duty", "detachment", "identity", "ego", "karma", "faith"
    ]
    base_reflections = {
        "fear": "Am I acting from fear or from faith?",
        "attachment": "What attachment is clouding my clarity?",
        "purpose": "Is my choice aligned with my soul's deeper purpose?",
        "duty": "Is this aligned with my dharma, not just my desires?",
        "detachment": "Can I act without attachment to results?",
        "identity": "Am I confusing my true self with my worldly role?",
        "ego": "Is my pride making this decision harder?",
        "karma": "What karmic seeds will I plant by this action?",
        "faith": "Am I doubting because I have lost faith in my path?"
    }
    selected_themes = random.sample(themes, 3)
    reflections = [base_reflections[theme] for theme in selected_themes]
    return reflections

# Streamlit UI — 🔥 Updated to Hide Gemini, Vyasa, Mirror, Technical
if streamlit_available:
    st.title("🪔 DharmaAI – Minimum Viable Conscience")
    st.subheader("Ask a question to GitaBot")

    available_modes = ["Krishna", "Krishna-Explains", "Arjuna"]  # Only showing Krishna, Krishna-Explains, Arjuna
    mode = st.sidebar.radio("Select Mode", available_modes)

    user_input = st.text_input("Your ethical question or dilemma:", value="")

    # Load verse matrix
    matrix_paths = [
        "data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
        "app/data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
        "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"
    ]
    df_matrix = None
    for path in matrix_paths:
        if os.path.exists(path):
            try:
                df_matrix = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                df_matrix = pd.read_csv(path, encoding='ISO-8859-1')
            break

    if st.button("🔍 Submit"):
        try:
            response = generate_gita_response(mode, df_matrix=df_matrix, user_input=user_input)
            st.markdown("""
            <div style='border: 1px solid #ddd; padding: 1rem; border-radius: 0.5rem; background-color: #f9f9f9;'>
            """, unsafe_allow_html=True)
            st.markdown(response)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

    if "Usage Journal" in st.session_state and st.session_state["Usage Journal"]:
        with st.expander("🕰️ View Past Interactions"):
            st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))
