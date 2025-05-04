import numpy as np

def get_embedding(text):
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(1536)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
