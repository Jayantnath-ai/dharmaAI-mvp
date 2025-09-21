# app/ui.py
from __future__ import annotations
import os
import logging
from pathlib import Path
import pandas as pd
import streamlit as st

from engine import ranking
from engine.planner import generate_action_plan, krishna_teaching

# -------- Config & logging --------
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dharmaai.ui")

st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")
st.title("🪔 DharmaAI – Minimum Viable Conscience")
st.subheader("Ask a question to GitaBot")

# -------- Cache heavy things --------
@st.cache_resource(show_spinner=False)
def load_matrix() -> pd.DataFrame | None:
    candidates = [
        Path(PROJECT_ROOT) / "data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
        Path(PROJECT_ROOT) / "app" / "data" / "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
        Path(PROJECT_ROOT) / "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                return pd.read_csv(p, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(p, encoding="ISO-8859-1")
    return None

df = load_matrix()
if df is None:
    st.error("⚠️ Could not load verse matrix CSV. Place it under /data/.")
    st.stop()

# -------- Modes --------
MODES = [
    "Krishna",
    "Krishna-Explains",
    "Arjuna",
    "Dharma Mirror",
    "Vyasa",
    "Technical",
    "Karmic Entanglement Simulator",
    "Forked Fate Contemplation"
]
mode = st.sidebar.radio("Select Mode", MODES)
user_input = st.text_input("Your ethical question or dilemma:", value="")

if "journal" not in st.session_state:
    st.session_state["journal"] = []

def _word_overlap(user_input: str, verse_text: str) -> str:
    u = set([w for w in (user_input or "").lower().split() if len(w) > 3])
    v = set([w for w in (verse_text or "").lower().split() if len(w) > 3])
    ov = list(u.intersection(v))
    return ", ".join(sorted(ov)[:3]) or "core theme alignment"

# -------- Submit --------
if st.button("🔍 Submit") and user_input.strip():
    try:
        top, row, col_text, col_id, col_tag = ranking.rank(df, user_input, top_k=3)
        verse_text = str(row[col_text])
        verse_id   = str(row[col_id]) if col_id else "—"
        verse_tag  = str(row[col_tag]) if col_tag else "detachment"

        st.markdown(f"""
**Nearest Verse:** `{verse_id}`  
*{verse_text}*  
_Tag:_ `{verse_tag}`
""")

        # ---- Mode rendering (kept simple; can be extended) ----
        if mode == "Krishna":
            why = _word_overlap(user_input, verse_text)
            st.markdown(f"""
**Krishna's Counsel**  
{krishna_teaching(verse_tag)}

**Why this verse?** Matched on **{verse_tag}** via **{why}**.
""")

        elif mode == "Krishna-Explains":
            plan = generate_action_plan(user_input, verse_tag)
            st.markdown(f"""
**Krishna's Teaching — Explained**  
{krishna_teaching(verse_tag)}

**Action Plan**

**Short Term (today–this week)**  
- {"\n- ".join(plan["short"])}

**Medium Term (next 2–6 weeks)**  
- {"\n- ".join(plan["medium"])}

**Long Term (quarter and beyond)**  
- {"\n- ".join(plan["long"])}
""")

        elif mode == "Technical":
            cols = [c for c in ["similarity", "similarity_tfidf", col_id, col_tag, col_text] if c in top.columns]
            st.code(top[cols].to_string(index=False), language="text")

        elif mode == "Karmic Entanglement Simulator":
            # toy two-path scoring
            from engine.planner import extract_signals
            sig = extract_signals(user_input)
            paths = [
                {"name":"Act Now","score":1.0 + (0.6 if sig["urgency"] else 0.0),"note":"Seize momentum with a reversible step."},
                {"name":"Wait & Verify","score":1.0 + (0.7 if sig["risk_words"] else 0.0) + (0.3 if sig["stakeholder_conflict"] else 0.0),"note":"Reduce harm via checks."}
            ]
            winner = sorted(paths, key=lambda x: x["score"], reverse=True)[0]
            st.markdown(f"""**Karmic Entanglement (Two-Path Simulation)**

- **Path A — {paths[0]['name']}** · score: {paths[0]['score']:.2f} · {paths[0]['note']}
- **Path B — {paths[1]['name']}** · score: {paths[1]['score']:.2f} · {paths[1]['note']}

**Mirror Verdict:** _Leaning **{winner['name']}**_ given current signals.
""")

        elif mode == "Arjuna":
            st.markdown("**Arjuna's Reflection**\n- Courage is clarity in motion; take one dharmic step now.")

        elif mode in ["Dharma Mirror", "Mirror"]:
            st.markdown("**Dharma Mirror**\n- Reflect with honesty: what is your true intention?\n- Choose the path that preserves dignity.")

        elif mode == "Vyasa":
            st.markdown("**Vyasa's Narration**\nYou are at a fork; name duties, forces, and attachments at play.")

        elif mode == "Forked Fate Contemplation":
            st.markdown("""**Forked Fate (Narrative)**
- **If you choose X:** You honor urgency but risk unseen harms. Keep the step reversible.
- **If you choose Y:** You reduce harm and bias, but momentum may fade. Timebox and revisit.
**Trade-off:** Seek a small, reversible probe that preserves dignity and learning while honoring duty.""")

        st.session_state["journal"].append({"q": user_input, "tag": verse_tag, "mode": mode})

    except Exception as e:
        logger.exception(e)
        st.error(f"Unexpected error: {e}")
