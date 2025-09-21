# engine/ranking.py
from __future__ import annotations
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Optional deps
try:
    from sentence_transformers import SentenceTransformer  # optional
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # optional
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# Optional helper module (if you provide your own embedding fn)
try:
    from utils.helpers import get_embedding as _get_embedding, cosine_similarity as _cosine_similarity
    HELPERS = True
except Exception:
    HELPERS = False

# ---------- Embeddings ----------
_model_cache = None
def _sbert(text: str):
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache.encode(text, convert_to_numpy=True)

def _fallback_embedding(text: str) -> np.ndarray:
    if HAS_SBERT:
        try:
            return _sbert(text)
        except Exception as e:
            logger.warning(f"SBERT unavailable: {e}")
    # deterministic pseudo-embedding as last resort
    rng = np.random.default_rng(abs(hash(text or 'default')) % (2**32))
    return rng.random(384)

def get_embedding(text: str) -> np.ndarray:
    if HELPERS:
        try:
            return _get_embedding(text)
        except Exception as e:
            logger.warning(f"helpers.get_embedding failed; fallback: {e}")
    return _fallback_embedding(text)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if HELPERS:
        try:
            return float(_cosine_similarity(v1, v2))
        except Exception as e:
            logger.warning(f"helpers.cosine_similarity failed; fallback: {e}")
    a, b = np.asarray(v1), np.asarray(v2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ---------- TF-IDF ----------
_tfidf = None
_tfidf_vecs = None
_tfidf_col = None

def _ensure_tfidf(df: pd.DataFrame, col_text: str):
    global _tfidf, _tfidf_vecs, _tfidf_col
    if not HAS_SKLEARN:
        return
    if _tfidf is not None and _tfidf_col == col_text:
        return
    texts = df[col_text].fillna("").astype(str).tolist()
    _tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000)
    _tfidf_vecs = _tfidf.fit_transform(texts)
    _tfidf_col = col_text

def tfidf_rank(df: pd.DataFrame, col_text: str, query: str, top_k: int = 3) -> pd.DataFrame | None:
    if not HAS_SKLEARN:
        return None
    _ensure_tfidf(df, col_text)
    qv = _tfidf.transform([query or ""])
    sims = (_tfidf_vecs @ qv.T).toarray().ravel()
    out = df.copy()
    out["similarity_tfidf"] = sims
    return out.sort_values("similarity_tfidf", ascending=False).head(top_k)

# ---------- Public API ----------
DEFAULT_TEXT_COLUMNS = ["Short English Translation", "English", "Verse", "Translation", "Summary"]
ID_COLUMNS = ["Verse ID", "ID", "Ref", "Key"]
TAG_COLUMNS = ["Symbolic Conscience Mapping", "Mapping", "Theme", "Tag"]

def find_text_col(df: pd.DataFrame) -> str | None:
    for c in DEFAULT_TEXT_COLUMNS:
        if c in df.columns:
            return c
    return None

def find_id_col(df: pd.DataFrame) -> str | None:
    for c in ID_COLUMNS:
        if c in df.columns:
            return c
    return None

def find_tag_col(df: pd.DataFrame) -> str | None:
    for c in TAG_COLUMNS:
        if c in df.columns:
            return c
    return None

def rank(df: pd.DataFrame, query: str, top_k: int = 3) -> tuple[pd.DataFrame, pd.Series, str, str, str]:
    """
    Returns (top_df, top_row, col_text, col_id, col_tag)
    Prefers TF-IDF when SBERT/helpers unavailable; otherwise embedding cosine.
    """
    if df is None or getattr(df, "empty", True):
        raise ValueError("Empty verse matrix")
    col_text = find_text_col(df)
    if not col_text:
        raise ValueError("No verse text column found")
    col_id = find_id_col(df)
    col_tag = find_tag_col(df)

    use_tfidf = (not HAS_SBERT and not HELPERS and HAS_SKLEARN)
    if use_tfidf:
        top = tfidf_rank(df, col_text, query, top_k=top_k)
        if top is None or top.empty:
            # last resort: embed fallback
            use_tfidf = False

    if not use_tfidf:
        if "embedding" not in df.columns:
            df = df.copy()
            df["embedding"] = df[col_text].fillna("default").apply(get_embedding)
        qv = get_embedding(query)
        df["similarity"] = df["embedding"].apply(lambda e: cosine_similarity(qv, e))
        if df["similarity"].isna().all() or df["similarity"].max() <= 0:
            # try TF-IDF once
            top = tfidf_rank(df, col_text, query, top_k=top_k)
            if top is None or top.empty:
                raise ValueError("No suitable matches; try rephrasing")
        else:
            top = df.sort_values("similarity", ascending=False).head(top_k)

    return top, top.iloc[0], col_text, col_id, col_tag
