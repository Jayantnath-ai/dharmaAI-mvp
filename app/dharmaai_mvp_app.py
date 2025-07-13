import sys
import os
import json
from datetime import datetime
from pathlib import Path
import logging

# 🔵 Set project root (modify as needed)
# Option 1: Dynamic path (assumes dharmaai_mvp_app.py is in project root)
project_root = str(Path(__file__).parent)
# Option 2: Hardcode absolute path (uncomment and set your path)
# project_root = "C:\\Users\\adminuser\\project"  # Windows
# project_root = "/home/adminuser/project"  # WSL
sys.path.append(project_root)
logger = logging.getLogger(__name__)
logger.info(f"Project root set to: {project_root}")
logger.info(f"sys.path updated: {sys.path}")

# 🔵 FEATURE FLAG: GitaBot integration
ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "true").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

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
    from components.modes import generate_arjuna_reflections, generate_dharma_mirror_reflections
    modes_available = True
except ImportError:
    modes_available = False
    logger.warning("components.modes not found; using fallback for Arjuna and Dharma Mirror modes")

# 🔵 Fallback functions for get_embedding and cosine_similarity
if not helpers_available:
    def get_embedding(text):
        if not text or not isinstance(text, str):
            text = "default"
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)

    def cosine_similarity(vec1, vec2):
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            logger.warning("Zero norm in cosine similarity")
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

# 🔵 MAIN GITA RESPONSE GENERATOR
def generate_gita_response(mode, df_matrix, user_input=None):
    if not user_input or len(user_input.strip()) < 5:
        logger.warning("Invalid user input provided")
        return "🛑 Please ask a more complete or meaningful question."

    if not pd or not np:
        logger.error("Pandas or NumPy not installed")
        return "⚠️ Error: Required libraries (pandas, numpy) not installed."

    if df_matrix is None or df_matrix.empty:
        logger.error("DataFrame is None or empty")
        return "⚠️ Error: Verse data not loaded. Please check the CSV file."

    # Validate required columns
    required_columns = ['Verse ID', 'Short English Translation', 'Symbolic Conscience Mapping']
    missing_columns = [col for col in required_columns if col not in df_matrix.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return f"⚠️ Error: Missing required columns in verse data: {missing_columns}"

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
        try:
            if 'embedding' not in df_matrix.columns:
                df_matrix['embedding'] = df_matrix['Short English Translation'].fillna("default").apply(get_embedding)
            user_embedding = get_embedding(user_input)
            df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))

            if df_matrix['similarity'].isna().all() or df_matrix['similarity'].max() == 0:
                logger.warning("No valid similarity scores computed")
                return "⚠️ Error: Unable to find a matching verse. Try rephrasing your question."

            verse_info = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]
        except Exception as e:
            logger.error(f"Error computing embeddings or similarity: {e}")
            return f"⚠️ Error computing embeddings or similarity: {str(e)}"

    if mode == "Krishna-Explains":
        if openai_available and os.getenv("OPENAI_API_KEY"):
            try:
                system_prompt = (
                    f"You are Krishna from the Bhagavad Gita. Provide dharma-aligned, symbolic, and contextual guidance. "
                    f"Verse context: '{verse_info['Short English Translation']}' with symbolic tag '{verse_info['Symbolic Conscience Mapping']}'"
                )
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
                response = (
                    f"**🤖 Krishna-Explains says:**\n\n"
                    f"_Reflecting on your question:_ **{user_input}**\n\n"
                    f"> {reply}"
                )
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                response = f"❌ Error fetching response from Krishna-Explains: {str(e)}"
        else:
            response = (
                f"**🤖 Krishna-Explains says:**\n\n"
                f"_Reflecting on your question:_ **{user_input}**\n\n"
                f"> {verse_info['Short English Translation'] if verse_info is not None else '[Simulated GPT response here based on dharma logic]'}"
            )
    elif mode == "Krishna":
        response = (
            f"**🧠 Krishna teaches:**\n\n"
            f"_You asked:_ **{user_input}**\n\n"
            f"> {verse_info['Short English Translation'] if verse_info is not None else '[Symbolic dharma insight would be offered here]'}"
        )
    elif mode == "Arjuna":
        if modes_available:
            try:
                reflections, matched_verse = generate_arjuna_reflections(user_input, df_matrix)
                reflection_text = "\n".join([f"{idx+1}. {line}" for idx, line in enumerate(reflections)])
                response = (
                    f"## 😟 Arjuna's Reflections\n\n"
                    f"_Reflecting on your question:_ **{user_input}**\n\n"
                    f"Here are three doubts arising in my mind:\n\n