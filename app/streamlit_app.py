
import sys
import random
import os
import re
import json
from datetime import datetime
import yaml
import pandas as pd

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

# Load scrolls (optional)
with open("data/ethical_scrolls.yaml") as f:
    scrolls_list = yaml.safe_load(f)
scrolls_df = pd.json_normalize(scrolls_list)

ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "true").lower() == "true"

def generate_dharma_mirror_reflections(user_input, df_matrix=None):
    traits = ["Service", "Non-attachment", "Devotion", "Non-malice", "Dharma Priority"]
    results = {
        trait: random.choice(["✅ Passed", "⚠️ Neutral", "❌ Violated"])
        for trait in traits
    }
    mirror_block = "**🪞 Mirror Reflection Results:**\n"
    for trait, status in results.items():
        mirror_block += f"- **{trait}**: {status}\n"
    return mirror_block

def generate_gita_response(mode, df_matrix, user_input=None):
    if mode == "Dharma Mirror":
        mirror_report = generate_dharma_mirror_reflections(user_input, df_matrix)
        return f"**Reflecting on your dilemma:**\n\n{user_input}\n\n" + mirror_report
    else:
        return f"🔧 Mode '{mode}' not implemented in this MVP."

def load_reflections(folder="saved_reflections"):
    return []

def analyze_reflections(reflections):
    return {}

def display_summary(summary):
    st.markdown("📝 No reflection summary available yet.")

if streamlit_available:
    st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")
    st.title("🪔 DharmaAI – Minimum Viable Conscience")

    st.subheader("Ask a question to GitaBot")

    if not ENABLE_GITABOT:
        st.warning("🔒 GitaBot integration is currently **disabled**. Please check back later.")
        st.stop()

    available_modes = ["Krishna", "Krishna-Explains", "Arjuna", "Dharma Mirror"]
    mode = st.sidebar.radio("Select Mode", available_modes)

    if st.sidebar.button("📊 Analyze Today's Reflections"):
        reflections = load_reflections()
        summary = analyze_reflections(reflections)
        display_summary(summary)

    user_input = st.text_input("Your ethical question or dilemma:", value="")

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

    if st.button("🔍 Submit") and user_input:
        try:
            response = generate_gita_response(mode, df_matrix=df_matrix, user_input=user_input)
            st.markdown(
                "<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>",
                unsafe_allow_html=True
            )
            st.markdown(response.replace("\n", "<br>"), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
