import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_dharma_mirror_reflections(user_input, df_matrix):
    # Validate inputs
    if not user_input or not isinstance(user_input, str) or len(user_input.strip()) < 5:
        logger.warning("Invalid user input provided")
        return ["Invalid input; please provide a meaningful question."], None
    
    if df_matrix is None or df_matrix.empty:
        logger.error("DataFrame is None or empty")
        return ["Error: Verse data not loaded. Please check the CSV file."], None

    required_columns = ['Short English Translation']
    missing_columns = [col for col in required_columns if col not in df_matrix.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return [f"Error: Missing required columns in verse data: {missing_columns}"], None

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

    try:
        # Compute embeddings and similarity
        user_embedding = get_embedding(user_input)
        df_matrix['embedding'] = df_matrix['Short English Translation'].fillna("default").apply(get_embedding)
        df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))

        if df_matrix['similarity'].isna().all() or df_matrix['similarity'].max() == 0:
            logger.warning("No valid similarity scores computed")
            return ["Error: Unable to find a matching verse. Try rephrasing your question."], None

        top_verse = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]
    except Exception as e:
        logger.error(f"Error in Dharma Mirror reflections: {e}")
        return [f"Error computing reflections: {str(e)}"], None

    reflections = [
        "🌿 What attachment are you still holding onto in this dilemma?",
        "🔥 Would your answer change if you were free from fear?",
        "🪞 Who does your choice serve — the self, others, or the dharma?",
        "💠 Is your question arising from confusion or a calling?",
        "⚖️ What karmic residue will this decision leave in the world?"
    ]

    return reflections, top_verse