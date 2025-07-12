import sys
import random
import os
import re
import json
from datetime import datetime

# 🔵 FEATURE FLAG: GitaBot integration
ENABLE_GITABOT = True  # Force enabled for testing
# ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "true").lower() == "true"
# ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "false").lower() == "false"
# ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "true").lower() == "false"



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

# 🔵 MAIN GITA RESPONSE GENERATOR

import numpy as np

def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or not isinstance(user_input, str) or len(user_input.strip()) < 5:
        return "🛑 Please enter a meaningful question or ethical dilemma."

    def get_embedding(text):
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    user_embedding = get_embedding(user_input)
    df_matrix['embedding'] = df_matrix['Short English Translation'].fillna("").apply(lambda x: get_embedding(x))
    df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))

    top_match = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]
    verse_id = top_match['Verse ID']
    verse_text = top_match['Short English Translation']
    symbolic_tag = top_match['Symbolic Conscience Mapping']
    ethical_trait = top_match['Ethical Trait']

    if mode == "Krishna":
        return f"**🧠 Krishna speaks (Verse {verse_id}):**

> _{verse_text}_

Symbolic Tag: **{symbolic_tag}** | Trait: _{ethical_trait}_"

    elif mode == "Arjuna":
        return f"**😟 Arjuna reflects:**

_I don't know what to do about_ **{user_input}**.

Let us reflect on verse {verse_id}: _{verse_text}_"

    elif mode == "Vyasa":
        return f"**📖 Vyasa narrates:**

When asked _'{user_input}'_, this verse arose:

> {verse_text}

Symbolic Mapping: {symbolic_tag}"

    elif mode == "Mirror":
        return f"> _You are not here to get an answer. You are here to remember your own dharma._"

    else:
        return f"**💬 GitaBot says:** {verse_text}"


def generate_arjuna_reflections(user_input, df_matrix):
    # … (unchanged) …
    pass

def generate_dharma_mirror_reflections(user_input, df_matrix):
    # … (unchanged) …
    pass

# 🔵 DAILY ANALYZER
def load_reflections(folder="saved_reflections"):
    # … (unchanged) …
    pass

def analyze_reflections(reflections):
    # … (unchanged) …
    pass

def display_summary(summary):
    # … (unchanged) …
    pass

# 🔵 STREAMLIT UI
if streamlit_available:
    st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")
    st.title("🪔 DharmaAI – Minimum Viable Conscience")

    # Always show core Conscience Layer
    st.subheader("Ask a question to GitaBot")

    # If the feature flag is OFF, show a notice and disable inputs
    if not ENABLE_GITABOT:
        st.warning("🔒 GitaBot integration is currently **disabled**. Please check back later.")
        st.stop()

    # ——— GitaBot-enabled UI ———
    available_modes = ["Krishna", "Krishna-Explains", "Arjuna", "Dharma Mirror"]
    mode = st.sidebar.radio("Select Mode", available_modes)

    if st.sidebar.button("📊 Analyze Today's Reflections"):
        reflections = load_reflections()
        summary = analyze_reflections(reflections)
        display_summary(summary)

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
            st.markdown(
                "<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>",
                unsafe_allow_html=True
            )
            st.markdown(response, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

    if "Usage Journal" in st.session_state and st.session_state["Usage Journal"]:
        with st.expander("🕰️ View Past Interactions"):
            st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))
