import streamlit as st
from components.gita_response import generate_gita_response
from components.analyzer import display_summary, load_reflections, analyze_reflections
import pandas as pd
import os

st.set_page_config(page_title="🪔 DharmaAI – GitaBot", layout="centered")
st.title("🪔 DharmaAI – Minimum Viable Conscience")
st.subheader("Ask a question to GitaBot")

# Load Matrix
df_matrix = None
matrix_paths = [
    "data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
    "app/data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
    "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"
]
for path in matrix_paths:
    if os.path.exists(path):
        df_matrix = pd.read_csv(path)
        break

# UI
mode = st.sidebar.radio("Select Mode", ["Krishna", "Arjuna", "Dharma Mirror", "Karmic Entanglement Simulator"])

if st.sidebar.button("📊 Analyze Today's Reflections"):
    summary = analyze_reflections(load_reflections())
    display_summary(summary)

user_input = st.text_input("Your ethical question or dilemma:", value="")

if st.button("🔍 Submit"):
    try:
        response = generate_gita_response(mode, df_matrix, user_input)
        st.markdown("<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>", unsafe_allow_html=True)
        st.markdown(response, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")

if "Usage Journal" in st.session_state and st.session_state["Usage Journal"]:
    with st.expander("🕰️ View Past Interactions"):
        st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))
