import streamlit as st
st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")

import os
import pandas as pd
from components.gita_response import generate_gita_response
from components.analyzer import display_summary, load_reflections, analyze_reflections

st.title("🪔 DharmaAI – Minimum Viable Conscience")
st.subheader("Ask a question to GitaBot")

available_modes = [
    "Krishna",
    "Krishna-Explains",
    "Arjuna",
    "Dharma Mirror",
    "Karmic Entanglement Simulator",
    "Vyasa",
    "Technical",
    "Forked Fate Contemplation"
]
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

if st.button("🔍 Submit"):
    try:
        response = generate_gita_response(mode, df_matrix=df_matrix, user_input=user_input)
        st.markdown("<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>", unsafe_allow_html=True)
        st.markdown(response, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")

if "Usage Journal" in st.session_state and st.session_state["Usage Journal"]:
    with st.expander("🕰️ View Past Interactions"):
        st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))
