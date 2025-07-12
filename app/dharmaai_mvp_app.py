import sys
import os
from pathlib import Path
# Add project root to Python path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

import random
import json
from datetime import datetime
import logging
import numpy as np
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 🔵 FEATURE FLAG: GitaBot integration
ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "true").lower() == "true"

# 🔵 MAIN GITA RESPONSE GENERATOR
def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or not isinstance(user_input, str) or len(user_input.strip()) < 5:
        logger.warning("Invalid user input provided")
        return "🛑 Please enter a meaningful question or ethical dilemma (at least 5 characters)."

    if df_matrix is None or df_matrix.empty:
        logger.error("DataFrame is None or empty")
        return "⚠️ Error: Verse data not loaded. Please check the CSV file."

    # Validate required columns
    required_columns = ['Verse ID', 'Short English Translation', 'Symbolic Conscience Mapping', 'Ethical Trait']
    missing_columns = [col for col in required_columns if col not in df_matrix.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return f"⚠️ Error: Missing required columns in verse data: {missing_columns}"

    def get_embedding(text):
        if not text or not isinstance(text, str):
            text = "default"
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)

    def cosine_similarity(vec1, vec2):
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    try:
        user_embedding = get_embedding(user_input)
        df_matrix['embedding'] = df_matrix['Short English Translation'].fillna("default").apply(get_embedding)
        df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))

        if df_matrix['similarity'].isna().all() or df_matrix['similarity'].max() == 0:
            logger.warning("No valid similarity scores computed")
            return "⚠️ Error: Unable to find a matching verse. Try rephrasing your question."

        top_match = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]
        verse_id = top_match['Verse ID']
        verse_text = top_match['Short English Translation'] or "No translation available"
        symbolic_tag = top_match['Symbolic Conscience Mapping'] or "None"
        ethical_trait = top_match['Ethical Trait'] or "None"

        # Log the response details
        logger.info(f"Generated response for mode '{mode}', verse ID: {verse_id}, similarity: {top_match['similarity']}")

        if mode == "Krishna":
            return f"**🧠 Krishna speaks (Verse {verse_id}):**\n\n> _{verse_text}_\n\nSymbolic Tag: **{symbolic_tag}** | Trait: _{ethical_trait}_"
        elif mode == "Arjuna":
            return f"**😟 Arjuna reflects:**\n\n_I don't know what to do about_ **{user_input}**.\n\nLet us reflect on verse {verse_id}: _{verse_text}_"
        elif mode == "Vyasa":
            return f"**📖 Vyasa narrates:**\n\nWhen asked _'{user_input}'_, this verse arose:\n\n> {verse_text}\n\nSymbolic Mapping: {symbolic_tag}"
        elif mode == "Dharma Mirror":
            return f"> _You are not here to get an answer. You are here to remember your own dharma._\n\nVerse {verse_id}: {verse_text}"
        else:
            return f"**💬 GitaBot says:** {verse_text}"
    except Exception as e:
        logger.error(f"Error in generate_gita_response: {e}")
        return f"⚠️ Error generating response: {e}"

def generate_arjuna_reflections(user_input, df_matrix):
    # Placeholder for unchanged function
    pass

def generate_dharma_mirror_reflections(user_input, df_matrix):
    # Placeholder for unchanged function
    pass

# 🔵 DAILY ANALYZER
def load_reflections(folder="saved_reflections"):
    # Placeholder for unchanged function
    pass

def analyze_reflections(reflections):
    # Placeholder for unchanged function
    pass

def display_summary(summary):
    # Placeholder for unchanged function
    pass

# 🔵 STREAMLIT UI
if streamlit_available:
    st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")
    st.title("🪔 DharmaAI – Minimum Viable Conscience")

    # Initialize session state
    if "Usage Journal" not in st.session_state:
        st.session_state["Usage Journal"] = []

    # Always show core Conscience Layer
    st.subheader("Ask a question to GitaBot")

    # If the feature flag is OFF, show a notice and disable inputs
    if not ENABLE_GITABOT:
        st.warning("🔒 GitaBot integration is currently **disabled**. Please check back later.")
        st.stop()

    # ——— GitaBot-enabled UI ———
    available_modes = ["Krishna", "Arjuna", "Vyasa", "Dharma Mirror"]
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
                logger.info(f"Loaded verse matrix from {path}")
                break
            except UnicodeDecodeError:
                df_matrix = pd.read_csv(path, encoding='ISO-8859-1')
                logger.info(f"Loaded verse matrix from {path} with ISO-8859-1 encoding")
                break
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
    if df_matrix is None:
        st.error("⚠️ Error: Could not load verse matrix CSV file. Please check the file path.")
        st.stop()

    if st.button("🔍 Submit"):
        try:
            response = generate_gita_response(mode, df_matrix=df_matrix, user_input=user_input)
            st.markdown(
                "<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>",
                unsafe_allow_html=True
            )
            st.markdown(response, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            # Log interaction to session state
            st.session_state["Usage Journal"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "input": user_input,
                "response": response
            })
        except Exception as e:
            logger.error(f"Streamlit UI error: {e}")
            st.error(f"⚠️ Unexpected error: {e}")

    if st.session_state["Usage Journal"]:
        with st.expander("🕰️ View Past Interactions"):
            st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))
else:
    logger.error("Streamlit is not available. Please install streamlit to run the UI.")
    print("Streamlit is not available. Please install streamlit to run the UI.")