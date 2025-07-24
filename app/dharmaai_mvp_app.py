import sys
import os
import json
from datetime import datetime
from pathlib import Path
import logging

# 🔵 Set project root
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

# Attempt imports
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

# Fallbacks for optional modules
try:
    from utils.helpers import get_embedding, cosine_similarity
    helpers_available = True
except ImportError:
    helpers_available = False
    logger.warning("utils.helpers not found; using fallback embedding functions")

try:
    from utils.dharma_mirror_utils import generate_dharma_mirror_reflections
    dharma_mirror_utils_available = True
except ImportError:
    dharma_mirror_utils_available = False
    logger.warning("utils.dharma_mirror_utils not found; using fallback reflections")

try:
    from components.modes import generate_arjuna_reflections
    modes_available = True
except ImportError:
    modes_available = False
    logger.warning("components.modes not found; using fallback modes")

try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    logger.warning("sentence_transformers not found; embeddings unavailable")

# Fallback embedding functions if needed
def get_embedding(text):
    if helpers_available:
        return get_embedding(text)
    if sentence_transformers_available:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
    # random fallback
    if np:
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)
    return []

def cosine_similarity(vec1, vec2):
    if helpers_available:
        return cosine_similarity(vec1, vec2)
    if not vec1 or not vec2:
        return 0.0
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

# Fallback Dharma Mirror reflections
if not dharma_mirror_utils_available:
    def generate_dharma_mirror_reflections(user_input, df_matrix):
        logger.warning("Using fallback reflections")
        return ["Reflect deeply."], None

# Main Gita response generator
def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or len(user_input.strip()) < 5:
        return "🛑 Please ask a more complete or meaningful question.", None
    if df_matrix is None or df_matrix.empty:
        return "⚠️ Error loading verse data.", None
    # Compute embeddings and similarity
    if 'embedding' not in df_matrix.columns:
        df_matrix['embedding'] = df_matrix['Short English Translation'].fillna("").apply(get_embedding)
    user_emb = get_embedding(user_input)
    df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_emb, emb))
    if df_matrix['similarity'].max() == 0:
        return "⚠️ No relevant verse found.", None
    top = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]
    # Simplified response assembly
    return f"Response based on verse {top['Verse ID']}: {top['Short English Translation']}", top

# Streamlit UI
if streamlit_available:
    st.set_page_config(page_title="🪔 DharmaAI – GitaBot", layout="centered")
    st.title("🪔 DharmaAI – GitaBot")
    # Ensure session state
    if "Usage Journal" not in st.session_state:
        st.session_state["Usage Journal"] = []
    # Feature toggle
    if not ENABLE_GITABOT:
        st.warning("GitaBot disabled.")
        st.stop()
    # Mode and input
    mode = st.sidebar.radio("Mode", ["Krishna", "Arjuna", "Vyasa", "Chorus"])
    user_input = st.text_input("Your question:")
    if st.button("Send") and user_input:
        # Load verse matrix
        import pandas as _pd
        df = _pd.read_csv(
            os.path.join(project_root, "data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"),
            encoding='utf-8'
        )
        response, verse = generate_gita_response(mode, df, user_input)
        st.markdown("**GitaBot Response:**")
        st.write(response)
        # Log usage
        st.session_state["Usage Journal"].append({"input": user_input, "response": response})
    # Display past interactions if any
    if st.session_state["Usage Journal"]:
        with st.expander("🕰️ Past Interactions"):
            st.dataframe(pd.DataFrame(st.session_state["Usage Journal"]))
else:
    print("Error: Streamlit not installed. Please install streamlit to run the UI.")
