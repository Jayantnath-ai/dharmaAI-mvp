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

# --- Optional imports with guards ---
try:
    import streamlit as st
    STREAMLIT = True
except Exception as e:
    STREAMLIT = False
    st = None
    logger.error("Streamlit is not available. Install streamlit to run the UI.")

try:
    import pandas as pd
    import numpy as np
    PANDAS_NUMPY = True
except Exception as e:
    pd = None
    np = None
    PANDAS_NUMPY = False
    logger.error("pandas/numpy not available.")

# Embeddings support (optional)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception as e:
    HAS_SBERT = False
    logger.warning("sentence_transformers not found; Sentence-BERT embeddings unavailable")

# Helpers (optional, we fallback if missing)
try:
    from utils.helpers import get_embedding as _get_embedding, cosine_similarity as _cosine_similarity
    HELPERS = True
except Exception as e:
    HELPERS = False
    logger.warning("utils.helpers not found; using fallback get_embedding and cosine_similarity")

# Dharma Mirror (optional)
try:
    from utils.dharma_mirror_utils import generate_dharma_mirror_reflections as _mirror_reflections
    HAS_MIRROR = True
except Exception as e:
    HAS_MIRROR = False
    logger.warning("utils.dharma_mirror_utils not found; using fallback for Dharma Mirror reflections")

# Modes (optional)
try:
    from components.modes import generate_arjuna_reflections as _arjuna_reflections
    HAS_MODES = True
except Exception as e:
    HAS_MODES = False
    logger.warning("components.modes not found; using fallback for Arjuna reflections")

# ---------- Fallback utilities ----------
def _fallback_embedding(text: str):
    if not PANDAS_NUMPY:
        return None
    if not text or not isinstance(text, str):
        text = "default"
    if HAS_SBERT:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error loading Sentence-BERT model: {e}")
    # Fallback to deterministic pseudo-random vector to keep UX stable
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.random(384)

def _fallback_cosine(a, b):
    if not PANDAS_NUMPY:
        return 0.0
    a = np.asarray(a); b = np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# Public wrappers used by the app
def get_embedding(text):
    if HELPERS:
        try:
            return _get_embedding(text)
        except Exception as e:
            logger.warning(f"helper.get_embedding failed, using fallback: {e}")
    return _fallback_embedding(text)

def cosine_similarity(v1, v2):
    if HELPERS:
        try:
            return _cosine_similarity(v1, v2)
        except Exception as e:
            logger.warning(f"helper.cosine_similarity failed, using fallback: {e}")
    return _fallback_cosine(v1, v2)

def mirror_reflections(user_input, df_matrix):
    if HAS_MIRROR:
        try:
            return _mirror_reflections(user_input, df_matrix)
        except Exception as e:
            logger.warning(f"mirror reflections failed, using fallback: {e}")
    # Fallback
    return [
        "Reflect with honesty: what is your true intention?",
        "Consider long-term outcomes over short-term gains.",
        "Choose the path that preserves dignity—yours and others'."
    ], None

def arjuna_reflections(user_input, df_matrix):
    if HAS_MODES:
        try:
            return _arjuna_reflections(user_input, df_matrix)
        except Exception as e:
            logger.warning(f"arjuna reflections failed, using fallback: {e}")
    return [
        "Your reluctance signals attachment—acknowledge it without judgment.",
        "Duty feels heavy when desire leads; align action with higher purpose.",
        "Courage is clarity in motion; take one dharmic step now."
    ]

# ---------- Core response generator (FIXED) ----------
def generate_gita_response(mode, df_matrix, user_input=None, top_k=3):
    \"\"\"Return (response_markdown, top_verse_row) with robust fallbacks.\"\"\"
    if not STREAMLIT:
        logger.error("Streamlit unavailable; cannot render UI.")
    if not user_input or len(user_input.strip()) < 3:
        return "🛑 Please ask a more complete or meaningful question.", None
    if not PANDAS_NUMPY:
        return "⚠️ Error: Required libraries (pandas, numpy) not installed.", None
    if df_matrix is None or getattr(df_matrix, 'empty', True):
        return "⚠️ Error: Verse data not loaded. Please check the CSV file.", None

    # Validate columns with leniency: support multiple common schemas
    # Pr
