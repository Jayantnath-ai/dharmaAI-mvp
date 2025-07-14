import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    logger.warning("sentence_transformers not found; using fallback random embeddings")

# Initialize Sentence-BERT model if available
model = None
if sentence_transformers_available:
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded Sentence-BERT model: all-MiniLM-L6-v2")
    except Exception as e:
        sentence_transformers_available = False
        logger.error(f"Failed to load Sentence-BERT model: {e}")

def get_embedding(text):
    if not text or not isinstance(text, str):
        text = "default"
    
    if sentence_transformers_available and model is not None:
        try:
            return model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error generating Sentence-BERT embedding: {e}")
            # Fall back to random embedding
    # Fallback to random embedding
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(1536)

def cosine_similarity(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        logger.warning("Zero norm in cosine similarity")
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)
