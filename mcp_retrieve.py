import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model and index once\ nmodel = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('context/faiss.index')

# scroll_texts: list of loaded scroll strings
def get_relevant(scroll_texts, query, k=5):
    q_vec = model.encode([query])
    D, I = index.search(np.array(q_vec).astype('float32'), k)
    return [scroll_texts[i] for i in I[0]]