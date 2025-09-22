# app/dharmaai_mvp_app.py
# 🪔 DharmaAI MVP — Cloud-Optimized & Context-Aware Single File
# - TF-IDF semantics (no torch downloads)
# - Dilemma detection, stakeholder/constraint extraction
# - Tag synthesis + boosted verse ranking
# - Krishna counsel + two-path fork + tailored plan

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("dharmaai.cloud")

# ---------------- Config ----------------
st.set_page_config(page_title="🪔 DharmaAI – GitaBot (Context-Aware)", layout="centered")
st.title("🪔 DharmaAI – Minimum Viable Conscience (Context-Aware, Cloud Optimized)")
st.caption("Contextualizes your dilemma → finds a verse → explains trade-offs → gives an actionable plan.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"

TEXT_COLUMNS = ["Short English Translation", "English", "Verse", "Translation", "Summary"]
ID_COLUMNS   = ["Verse ID", "ID", "Ref", "Key"]
TAG_COLUMNS  = ["Symbolic Conscience Mapping", "Mapping", "Theme", "Tag"]

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

# ---------------- Column helpers ----------------
def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------------- NLP-ish utilities (lightweight) ----------------
def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']+", (s or "").lower())

def extract_stakeholders(text: str) -> List[str]:
    # Simple dictionary; adjust as needed
    candidates = {
        "customer","user","client","investor","board","team","manager","engineer","designer","legal",
        "compliance","security","auditor","vendor","partner","regulator","family","spouse","child","students",
        "community","public"
    }
    toks = set(_tokenize(text))
    return sorted([w for w in candidates if w in toks])

def extract_constraints(text: str) -> List[str]:
    patterns = [
        (r"\b(deadline|friday|today|tomorrow|asap|q[1-4]|quarter|sprint)\b", "time"),
        (r"\b(budget|cost|revenue|profit|roi|runway|funding|payroll)\b", "finance"),
        (r"\b(policy|law|legal|regulation|regulatory|license|contract|nda)\b", "legal"),
        (r"\b(risk|privacy|security|safety|breach|fraud|bias|fairness)\b", "risk"),
        (r"\b(prod|production|ship|release|deploy|oncall|severe|sev)\b", "delivery")
    ]
    found = []
    for patt, tag in patterns:
        if re.search(patt, (text or "").lower()):
            found.append(tag)
    return sorted(set(found))

def extract_signals(text: str) -> Dict[str, bool]:
    t = (text or "").lower()
    return {
        "urgency": any(k in t for k in ["urgent","deadline","now","asap","today","tonight","tomorrow","friday"]),
        "uncertainty": any(k in t for k in ["uncertain","unknown","ambiguous","confused","unsure","dilemma","conflict"]),
        "stakeholder_conflict": any(k in t for k in ["team","manager","customer","client","board","stakeholder","investor","legal","compliance","security"]),
        "risk_words": any(k in t for k in ["risk","harm","unsafe","privacy","bias","security","safety","breach","fraud"]),
    }

def detect_dilemma(text: str) -> str:
    t = (text or "").lower()
    # Very small rules; extend as needed
    if any(k in t for k in ["tell the truth","hide","conceal","whistleblow","disclose","cover up","transparency","lie"]):
        return "truth_vs_loyalty"
    if any(k in t for k in ["deadline","ship","deliver","kpi","target"]) and any(k in t for k in ["harm","safety","privacy","bias"]):
        return "duty_vs_harm"
    if any(k in t for k in ["unfair","bias","discriminate","favor","prefer"]):
        return "fairness_vs_utility"
    if any(k in t for k in ["attach","fear","status","approval","anxiety","ego","desire"]):
        return "self_control"
    if any(k in t for k in ["pivot","change","uncertain","volatile","unstable","impermanent"]):
        return "impermanence"
    # default: duty dilemma if conflicts & delivery show up
    if any(k in t for k in ["conflict","disagree","pushback"]) and any(k in t for k in ["ship","deliver","deadline"]):
        return "duty_vs_outcome"
    return "general"

def synthesize_tag(question: str, verse_text_hint: str = "") -> str:
    t = f"{question} {verse_text_hint}".lower()
    if any(k in t for k in ["duty","role","obligation","responsibility"]): return "duty"
    if any(k in t for k in ["harm","compassion","ahimsa","kindness","non-violence"]): return "compassion"
    if any(k in t for k in ["truth","honest","transparency","satya","whistleblow","disclose"]): return "truth"
    if any(k in t for k in ["discipline","self","control","temperance","ego","desire","attachment"]): return "self-control"
    if any(k in t for k in ["fair","bias","discriminate","equity","justice"]): return "fairness"
    if any(k in t for k in ["impermanence","change","uncertain","volatile","entropy"]): return "impermanence"
    return "detachment"

def krishna_teaching(tag: str) -> str:
    tag = (tag or "").lower()
    if "duty" in tag:           return "Right action honors your role without vanity or avoidance."
    if "compassion" in tag:     return "Choose paths that reduce harm and preserve dignity."
    if "truth" in tag:          return "Clarity grows where truth is chosen over convenience."
    if "self" in tag or "discipline" in tag: return "Mastery of self calms the storm before action."
    if "fair" in tag:           return "Fairness aligns effort with justice; avoid partiality."
    if "impermanence" in tag:   return "Act wisely, knowing conditions change; keep options flexible."
    return "Detach from outcomes; align with your highest duty."

# ---------------- Ranking (TF-IDF + boosts) ----------------


def rank_verses(df: pd.DataFrame, query: str, col_text: str, col_id: str | None, col_tag: str | None, top_k=3):
    vectorizer, vecs = build_tfidf(df, col_text)
    qv = vectorizer.transform([query])
    sims = (vecs @ qv.T).toarray().ravel()

    # --- Boosts ---
    bonus = np.zeros_like(sims)

    # Boost 1: tag match to synthesized query tag
    q_tag = synthesize_tag(query)
    if q_tag and col_tag and col_tag in df.columns:
        tag_mask = df[col_tag].fillna("").astype(str).str.lower().str.contains(re.escape(q_tag), na=False).values
        bonus += tag_mask.astype(float) * 0.03

    # Boost 2: stakeholder word overlap
    stakeholders = extract_stakeholders(query)
    if stakeholders:
        stk = set(stakeholders)
        verse_tokens = [set(_tokenize(txt)) for txt in df[col_text].fillna("").astype(str).tolist()]
        stk_bonus = np.array([0.02 * len(stk.intersection(vt)) for vt in verse_tokens])
        bonus += stk_bonus

    score = sims + bonus
    out = df.copy()
    out["similarity"] = score
    top = out.sort_values("similarity", ascending=False).head(top_k)
    return top, top.iloc[0]


# ---------------- Plan generation ----------------
def generate_action_plan(user_input: str, tag: str | None, signals: Dict[str, bool], constraints: List[str]) -> Dict[str, List[str]]:
    short, medium, long = [], [], []

    # Base rules by signals
    if signals["risk_words"]:
        short.append("Run a harm scan: privacy, safety, bias, security.")
    if signals["urgency"]:
        short.append("Take one **reversible** step in the next 24 hours.")
    if signals["stakeholder_conflict"]:
        short.append("Clarify **decision rights** and who must be informed.")
    if signals["uncertainty"]:
        medium.append("Design a small **probe** to learn before committing.")
    if "legal" in constraints or "risk" in constraints:
        medium.append("Log a one-page **decision record** with assumptions and stop-rules.")

    # Tag-specific nudges
    t = (tag or "").lower()
    if "truth" in t:
        short.insert(0, "Surface the **inconvenient fact** you are tempted to hide.")
        medium.append("Plan a **transparent disclosure** with safety and dignity preserved.")
    elif "compassion" in t:
        short.insert(0, "List affected parties and **minimize harm** first.")
        medium.append("Add a **dignity check** to the approval path.")
    elif "duty" in t:
        short.insert(0, "Write your **non-negotiable duty** in one sentence.")
        medium.append("Share trade-offs **plainly** with stakeholders.")
    elif "fair" in t:
        short.insert(0, "Check for **bias** and hidden favoritism.")
        medium.append("Define **fairness criteria** you will measure.")
    elif "self" in t:
        short.insert(0, "Insert a **pause** (10 breaths) before you act.")
        medium.append("Create a **trigger** for calm escalation.")
    elif "impermanence" in t:
        short.insert(0, "Separate transient noise from **signal**.")
        medium.append("Prefer options that **keep flexibility**.")

    # Long-term hygiene
    long.extend([
        "Institutionalize a **recurring review** (pre/post decision).",
        "Codify the policy so it **survives handoffs**.",
        "Track **harm-reduction and fairness**, not only outcomes."
    ])

    # De-duplicate while preserving order
    def dedup(seq): 
        seen=set(); out=[]
        for x in seq:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return {
        "short": dedup(short)[:5] or ["Clarify duty; take one reversible step; name what you will measure."],
        "medium": dedup(medium)[:5] or ["Pilot with checkpoints; invite a counter-perspective; document trade-offs."],
        "long": dedup(long)[:5]
    }

# ---------------- Fork simulation ----------------
def simulate_two_paths(signals: Dict[str, bool]) -> Tuple[str, List[str]]:
    # Scores based on signals
    paths = [
        {"name":"Act Now", "score":1.0, "note":"Seize momentum with a **reversible** step."},
        {"name":"Wait & Verify", "score":1.0, "note":"Reduce harm with **checks** and clear stop-rules."}
    ]
    if signals["urgency"]:
        paths[0]["score"] += 0.6
    if signals["risk_words"]:
        paths[1]["score"] += 0.7
    if signals["stakeholder_conflict"]:
        paths[1]["score"] += 0.3

    winner = sorted(paths, key=lambda p: p["score"], reverse=True)[0]["name"]
    lines = [
        f"- **Path A — {paths[0]['name']}** · score: {paths[0]['score']:.2f} · {paths[0]['note']}",
        f"- **Path B — {paths[1]['name']}** · score: {paths[1]['score']:.2f} · {paths[1]['note']}",
        f"**Mirror Verdict:** Leaning **{winner}** given the current signals."
    ]
    return winner, lines

# ---------------- Core response ----------------
def generate_gita_response(mode: str, df: pd.DataFrame, user_input: str, top_k: int = 3) -> str:
    if not user_input.strip():
        return "🛑 Please ask a more complete or meaningful question."

    col_text = find_col(df, TEXT_COLUMNS)
    col_id   = find_col(df, ID_COLUMNS)
    col_tag  = find_col(df, TAG_COLUMNS)
    if not col_text:
        return "⚠️ Error: No verse text column found."

    # Extract context
    signals = extract_signals(user_input)
    stakeholders = extract_stakeholders(user_input)
    constraints  = extract_constraints(user_input)
    dilemma      = detect_dilemma(user_input)

    # Rank verses (TF-IDF + boosts)
    top, row = rank_verses(df, user_input, col_text, col_id, col_tag, top_k=top_k)
    verse_text = str(row[col_text])
    verse_id   = str(row[col_id]) if col_id else "—"
    verse_tag  = str(row[col_tag]) if col_tag and pd.notna(row[col_tag]) else synthesize_tag(user_input, verse_text)

    # Narrative header
    header = f"""
**Nearest Verse:** `{verse_id}`  
*{verse_text}*  
_Tag:_ `{verse_tag}` · _Dilemma:_ `{dilemma}`  
_Stakeholders:_ `{", ".join(stakeholders) if stakeholders else "—"}` · _Constraints:_ `{", ".join(constraints) if constraints else "—"}`
"""

    # Mode bodies
    if mode == "Krishna":
        body = f"""
**Krishna's Counsel**  
{krishna_teaching(verse_tag)}

**Why this verse?**  
- TF-IDF semantic match to your question  
- Boosted by `{verse_tag}` theme and stakeholder overlap
"""
    elif mode == "Krishna-Explains":
        plan = generate_action_plan(user_input, verse_tag, signals, constraints)
        body = f"""
**Krishna's Teaching — Explained**  
{krishna_teaching(verse_tag)}

**Two-Path Fork (Dharma Reflection)**  
""" + "\n".join(simulate_two_paths(signals)[1]) + f"""

**Action Plan**  
**Short (today–this week)**  
- {"\n- ".join(plan["short"])}

**Medium (2–6 weeks)**  
- {"\n- ".join(plan["medium"])}

**Long (quarter and beyond)**  
- {"\n- ".join(plan["long"])}
"""
    elif mode == "Technical":
        cols = [c for c in ["similarity", col_id, col_tag, col_text] if c and c in top.columns]
        body = f"""
**Technical Trace**  
Top-{top_k} (higher = closer):

"""
    else:
        # Concise “mirror” fallback
        winner, lines = simulate_two_paths(signals)
        body = "**Dharma Mirror**\n" + "\n".join(lines) + "\n\n" + krishna_teaching(verse_tag)

    footer = "\n---\n*Tip:* Name the duty, who decides, what harms you’ll avoid, and the smallest reversible step."
    return header + "\n\n" + body + "\n\n" + footer

# ---------------- UI ----------------
df = load_matrix()
if df is None:
    st.error("⚠️ Could not load verse matrix CSV. Place it under /data/.")
    st.stop()

MODES = ["Krishna", "Krishna-Explains", "Technical", "Mirror"]
mode = st.sidebar.radio("Select Mode", MODES)
user_input = st.text_input("Describe your ethical dilemma or decision:")

if st.button("🔍 Submit") and user_input.strip():
    response = generate_gita_response(mode, df, user_input)
    if response.startswith(("⚠️","🛑")):
        st.error(response)
    else:
        st.markdown(response)
