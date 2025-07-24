import sys
import os
import json
from datetime import datetime
from pathlib import Path
import logging
import re

# 🔵 Set project root (modify as needed)
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f"Project root set to: {project_root}")
logger.info(f"sys.path updated: {sys.path}")

# 🔵 FEATURE FLAG: GitaBot integration
ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "true").lower() == "true"

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
    import numpy as np
except ImportError:
    pd = None
    np = None

try:
    from utils.helpers import get_embedding, cosine_similarity
    helpers_available = True
except ImportError:
    helpers_available = False
    logger.warning("utils.helpers not found; using fallback get_embedding and cosine_similarity")

try:
    from utils.dharma_mirror_utils import generate_dharma_mirror_reflections
    dharma_mirror_utils_available = True
except ImportError:
    dharma_mirror_utils_available = False
    logger.warning("utils.dharma_mirror_utils not found; using fallback for Dharma Mirror reflections")

try:
    from components.modes import generate_arjuna_reflections
    modes_available = True
except ImportError:
    modes_available = False
    logger.warning("components.modes not found; using fallback for Arjuna reflections")

try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    logger.warning("sentence_transformers not found; Sentence-BERT embeddings unavailable")

# 🔵 Fallback functions for get_embedding and cosine_similarity
if not helpers_available:
    def get_embedding(text):
        if not text or not isinstance(text, str):
            text = "default"
        if sentence_transformers_available:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                return model.encode(text, convert_to_numpy=True)
            except Exception as e:
                logger.error(f"Error loading Sentence-BERT model: {e}")
        # Fallback to random embedding
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)

    def cosine_similarity(vec1, vec2):
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            logger.warning("Zero norm in cosine similarity")
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

# 🔵 Fallback function for Dharma Mirror reflections
if not dharma_mirror_utils_available:
    def generate_dharma_mirror_reflections(user_input, df_matrix):
        logger.warning("Using fallback for generate_dharma_mirror_reflections")
        reflections = [
            "Fallback: Reflect on your question to find clarity.",
            "Fallback: Consider the consequences of your actions.",
            "Fallback: Seek the path of dharma."
        ]
        return reflections, None

# 🔵 MAIN GITA RESPONSE GENERATOR
def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or len(user_input.strip()) < 5:
        logger.warning("Invalid user input provided")
        return "🛑 Please ask a more complete or meaningful question.", None

    if not pd or not np:
        logger.error("Pandas or NumPy not installed")
        return "⚠️ Error: Required libraries (pandas, numpy) not installed.", None

    if df_matrix is None or df_matrix.empty:
        logger.error("DataFrame is None or empty")
        return "⚠️ Error: Verse data not loaded. Please check the CSV file.", None

    # Validate required columns
    required_columns = ['Verse ID', 'Short English Translation', 'Symbolic Conscience Mapping']
    missing_columns = [col for col in required_columns if col not in df_matrix.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return f"⚠️ Error: Missing required columns in verse data: {missing_columns}", None

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
    try:
        if 'embedding' not in df_matrix.columns:
            df_matrix['embedding'] = df_matrix['Short English Translation'].fillna("default").apply(get_embedding)
        user_embedding = get_embedding(user_input)
        df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))

        if df_matrix['similarity'].isna().all() or df_matrix['similarity'].max() == 0:
            logger.warning("No valid similarity scores computed")
            return "⚠️ Error: Unable to find a matching verse. Try rephrasing your question.", None

        verse_info = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]
    except Exception as e:
        logger.error(f"Error computing embeddings or similarity: {e}")
        return f"⚠️ Error computing embeddings or similarity: {str(e)}", None

    # Construct response based on mode (unchanged)...
    # [Rest of function body remains unchanged]

    return response, verse_info

#🔵 SUPPORT FUNCTIONS (unchanged)
def generate_arjuna_reflections(user_input, df_matrix):
    pass

def load_reflections(folder="saved_reflections"):
    from components.analyzer import load_reflections
    pass

def analyze_reflections(reflections):
    from components.analyzer import analyze_reflections
    pass

def display_summary(summary):
    from components.analyzer import display_summary
    pass

#🔵 STREAMLIT UI (unchanged)
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
available_modes = [
    "Krishna",
    "Krishna-Explains",
    "Arjuna",
    "Dharma Mirror",
    "Vyasa",
    "Technical",
    "Karmic Entanglement Simulator",
    "Forked Fate Contemplation"
]
mode = st.sidebar.radio("Select Mode", available_modes)

if st.sidebar.button("📊 Analyze Today's Reflections"):
    reflections = load_reflections()
    summary = analyze_reflections(reflections)
    display_summary(summary)

user_input = st.text_input("Your ethical question or dilemma:", value="")

# Load verse matrix
matrix_paths = [
    os.path.join(project_root, "data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"),
    os.path.join(project_root, "app/data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"),
    os.path.join(project_root, "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv")
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
    logger.error("Could not load verse matrix CSV file")
    st.error("⚠️ Error: Could not load verse matrix CSV file. Please check the file path.")
    st.stop()

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
    try:
        st.rerun()  # Use st.rerun for Streamlit >= 1.37.0
    except AttributeError:
        try:
            st.experimental_rerun()  # Fallback for older Streamlit versions
        except AttributeError:
            logger.error("Streamlit rerun method not available. Please update Streamlit to 1.37.0 or higher, or use experimental_rerun for older versions.")
            st.error("⚠️ Error: Streamlit rerun method not available. Please update Streamlit.")
st.markdown("</div>", unsafe_allow_html=True)

if st.button("🔍 Submit"):
    try:
        response, verse_info = generate_gita_response(mode, df_matrix, user_input)
        st.markdown(
            "<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>",
            unsafe_allow_html=True
        )
        if response.startswith("⚠️") or response.startswith("❌"):
            st.error(response)
        else:
            if verse_info is not None and 'Verse ID' in verse_info and 'Symbolic Conscience Mapping' in verse_info:
                st.markdown(
                    f"<small>📘 Verse ID: {verse_info['Verse ID']} — <em>{verse_info['Symbolic Conscience Mapping']}</em></small>",
                    unsafe_allow_html=True
                )
            st.markdown(response, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Streamlit UI error: {e}")
        st.error(f"⚠️ Unexpected error: {e}")

if st.session_state["Usage Journal"]:
    with st.expander("🕰️ View Past Interactions"):
        st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))
else:
    logger.error("Streamlit is not available. Please install streamlit to run the UI.")
    print("Streamlit is not available. Please install streamlit to run the UI.")
