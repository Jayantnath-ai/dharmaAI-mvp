import pytest
from utils import helpers
import numpy as np

def test_get_embedding_shape():
    emb = helpers.get_embedding("test input")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1536,)

def test_cosine_similarity_range():
    vec1 = helpers.get_embedding("A")
    vec2 = helpers.get_embedding("B")
    score = helpers.cosine_similarity(vec1, vec2)
    assert 0 <= score <= 1
