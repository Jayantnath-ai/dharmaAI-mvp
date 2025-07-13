
import numpy as np

def generate_dharma_mirror_reflections(user_input, df_matrix):
    def get_embedding(text):
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(1536)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    user_embedding = get_embedding(user_input)
    df_matrix['embedding'] = df_matrix['Short English Translation'].fillna("").apply(get_embedding)
    df_matrix['similarity'] = df_matrix['embedding'].apply(lambda emb: cosine_similarity(user_embedding, emb))
    top_verse = df_matrix.sort_values(by='similarity', ascending=False).iloc[0]

    reflections = [
        "🌿 What attachment are you still holding onto in this dilemma?",
        "🔥 Would your answer change if you were free from fear?",
        "🪞 Who does your choice serve — the self, others, or the dharma?",
        "💠 Is your question arising from confusion or a calling?",
        "⚖️ What karmic residue will this decision leave in the world?"
    ]

    return reflections, top_verse
