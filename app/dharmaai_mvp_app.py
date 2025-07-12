import sys
import random
import os
import re
import json
from datetime import datetime

# 🔵 FEATURE FLAG: GitaBot integration
ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "true").lower() == "false"
# ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "false").lower() == "true"
# ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "true").lower() == "true"
# ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "false").lower() == "false"


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
def generate_gita_response(mode, df_matrix, user_input=None):
    # … [leave this function unchanged] …
    pass  # (omitted here for brevity)

# 🔵 SUPPORT FUNCTIONS
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
