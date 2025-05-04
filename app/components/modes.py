from utils.helpers import get_embedding, cosine_similarity
import re

def generate_arjuna_reflections(user_input, df_matrix):
    if df_matrix is None or df_matrix.empty:
        return [
            "Am I acting from fear or from faith?",
            "Is this choice aligned with my dharma or my desire?",
            "What attachment clouds my clarity?"
        ], "[No matched verse]"

    user_embedding = get_embedding(user_input)
    df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))
    best_match = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]
    symbolic_tag = best_match.get('Symbolic Conscience Mapping', 'Unknown Dharma Theme')
    verse_text = best_match.get('Short English Translation', '[Verse unavailable]')

    reflections = [
        f"How does {symbolic_tag} guide my next action?",
        f"Am I being tested in the realm of {symbolic_tag}?",
        f"What does {symbolic_tag} demand from me, not my fear?"
    ]
    return reflections, verse_text

def generate_dharma_mirror_reflections(user_input, df_matrix):
    translation_map = {
        "Atma-Sarvatra": "Soul Everywhere",
        "Karma-Yoga": "Path of Selfless Action",
        "Jnana-Yoga": "Path of Knowledge",
        "Bhakti-Yoga": "Path of Devotion",
    }

    if df_matrix is None or df_matrix.empty:
        return [
            "What is the silent truth behind my dilemma?",
            "Which attachment clouds my view?",
            "What would courage — not fear — choose here?"
        ], "[No matched verse]"

    user_embedding = get_embedding(user_input)
    df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))
    best_match = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]

    symbolic_tag = best_match.get('Symbolic Conscience Mapping', 'Unknown Dharma Theme')
    symbolic_tag = re.sub(r'Field Mode', '', symbolic_tag).strip()
    if symbolic_tag in translation_map:
        symbolic_tag = f"{symbolic_tag} ({translation_map[symbolic_tag]})"

    reflections = [
        f"What would {symbolic_tag} advise me to release?",
        f"In what way is {symbolic_tag} already guiding my path?",
        f"How would {symbolic_tag} shape my decision without fear or desire?"
    ]
    return reflections, best_match.get('Short English Translation', '[Verse unavailable]')
