# app/dharmaai_mvp_app.py
# 🪔 DharmaAI MVP – Cloud Optimized Single File Version
# Runs fast on Streamlit Cloud (no heavy ML downloads)

import os
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dharmaai.cloud")

# ---------------- Config ----------------
st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")
st.title("🪔 DharmaAI – Minimum Viable Conscience (Cloud Optimized)")
st.subheader("Ask a question to GitaBot")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"

# ---------------- Cache Heavy Resources ----------------
@st.cache_resource(show_spinner=False)
def load_matrix() -> pd.DataFrame | None:
    if DATA_FILE.exists():
        try:
            return pd.read_csv(DATA_FILE, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(DATA_FILE, encoding="ISO-8859-1")
    return None

@st.cache_resource(show_spinner=False)
def build_tfidf(df: pd.DataFrame, col_text: str):
    texts = df[col_text].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
    vecs = vectorizer.fit_transform(texts)
    return vectorizer, vecs

# ---------------- Utility ----------------
def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def extract_signals(text: str) -> dict:
    t = (text or "").lower()
    return {
        "urgency": any(k in t for k in ["urgent","deadline","now","asap","today"]),
        "uncertainty": any(k in t for k in ["uncertain","unknown","ambiguous","confused"]),
        "stakeholder_conflict": any(k in t for k in ["team","manager","customer","partner","board","investor","client","legal","compliance"]),
        "risk_words": any(k in t for k in ["risk","harm","unsafe","privacy","bias","security","safety","breach","fraud"]),
    }

def krishna_teaching(tag: str) -> str:
    tag = (tag or "").lower()
    if "duty" in tag:           return "Right action honors your role without vanity or avoidance."
    if "compassion" in tag:     return "Choose paths that reduce harm and preserve dignity."
    if "truth" in tag:          return "Clarity grows where truth is chosen over convenience."
    if "self" in tag or "discipline" in tag: return "Mastery of self calms the storm before action."
    if "impermanence" in tag:   return "Act wisely, knowing conditions change; keep options flexible."
    return "Detach from outcomes; align with your highest duty."

# ---------------- Core Logic ----------------
def generate_gita_response(mode: str, df: pd.DataFrame, user_input: str, top_k: int = 3):
    if not user_input.strip():
        return "🛑 Please ask a more complete or meaningful question."

    # --- Find relevant columns
    col_text = find_col(df, ["Short English Translation","English","Verse","Translation","Summary"])
    col_id   = find_col(df, ["Verse ID","ID","Ref","Key"])
    col_tag  = find_col(df, ["Symbolic Conscience Mapping","Mapping","Theme","Tag"])
    if not col_text:
        return "⚠️ Error: No verse text column found."

    # --- TF-IDF similarity
    vectorizer, vecs = build_tfidf(df, col_text)
    qv = vectorizer.transform([user_input])
    sims = (vecs @ qv.T).toarray().ravel()
    df = df.copy()
    df["similarity"] = sims
    top = df.sort_values("similarity", ascending=False).head(top_k)
    row = top.iloc[0]

    verse_text = str(row[col_text])
    verse_id   = str(row[col_id]) if col_id else "—"
    verse_tag  = str(row[col_tag]) if col_tag else "detachment"

    header = f"""
**Nearest Verse:** `{verse_id}`  
*{verse_text}*  
_Tag:_ `{verse_tag}`
"""

    # --- Mode responses
    if mode == "Krishna":
        body = f"""
**Krishna's Counsel**  
{krishna_teaching(verse_tag)}

**Why this verse?** Matched on **{verse_tag}** with TF-IDF overlap.
"""
    elif mode == "Krishna-Explains":
        sig = extract_signals(user_input)
        plan = []
        if sig["risk_words"]: plan.append("- Run a harm scan (privacy, safety, bias).")
        if sig["urgency"]: plan.append("- Take one reversible step today.")
        if sig["stakeholder_conflict"]: plan.append("- Clarify who has decision rights.")
        if not plan: plan = ["- Clarify your duty.", "- Act without clinging to outcomes."]
        body = f"""
**Krishna's Teaching — Explained**  
{krishna_teaching(verse_tag)}

**Action Plan**  
{chr(10).join(plan)}
"""
    elif mode == "Technical":
        cols = [c for c in ["similarity", col_id, col_tag, col_text] if c and c in top.columns]
        body = f"**Technical Trace**\n\n```\n{top[cols].to_string(index=False)}\n```"
    else:
        body = "Choose dharma; preserve dignity and long-term harmony."

    footer = "\n---\n*Tip:* Add people, constraints, and values you refuse to compromise."
    return header + "\n\n" + body + "\n\n" + footer

# ---------------- UI ----------------
df = load_matrix()
if df is None:
    st.error("⚠️ Could not load verse matrix CSV. Place it under /data/.")
    st.stop()

MODES = ["Krishna","Krishna-Explains","Technical"]
mode = st.sidebar.radio("Select Mode", MODES)
user_input = st.text_input("Your ethical question or dilemma:")

if st.button("🔍 Submit") and user_input.strip():
    response = generate_gita_response(mode, df, user_input)
    if response.startswith(("⚠️","🛑")):
        st.error(response)
    else:
        st.markdown(response)
