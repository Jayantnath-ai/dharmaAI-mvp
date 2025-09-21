import sys
import os
import logging
from pathlib import Path

# 🔵 Set project root (modify as needed)
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f"Project root set to: {project_root}")
logger.info(f"sys.path updated: {sys.path}")

# 🔵 FEATURE FLAG: GitaBot integration
ENABLE_GITABOT = os.getenv("ENABLE_GITABOT", "true").lower() == "true"

# --- Optional imports with guards ---
try:
    import streamlit as st
    STREAMLIT = True
    st.set_page_config(page_title="🪔 DharmaAI – GitaBot Reflection Engine", layout="centered")
except Exception:
    STREAMLIT = False
    st = None
    logger.error("Streamlit is not available. Install streamlit to run the UI.")

try:
    import pandas as pd
    import numpy as np
    PANDAS_NUMPY = True
except Exception:
    pd = None
    np = None
    PANDAS_NUMPY = False
    logger.error("pandas/numpy not available.")

# Embeddings support (optional)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False
    logger.warning("sentence_transformers not found; Sentence-BERT embeddings unavailable")

# Helpers (optional, we fallback if missing)
try:
    from utils.helpers import get_embedding as _get_embedding, cosine_similarity as _cosine_similarity
    HELPERS = True
except Exception:
    HELPERS = False
    logger.warning("utils.helpers not found; using fallback get_embedding and cosine_similarity")

# Dharma Mirror (optional)
try:
    from utils.dharma_mirror_utils import generate_dharma_mirror_reflections as _mirror_reflections
    HAS_MIRROR = True
except Exception:
    HAS_MIRROR = False
    logger.warning("utils.dharma_mirror_utils not found; using fallback for Dharma Mirror reflections")

# Modes (optional)
try:
    from components.modes import generate_arjuna_reflections as _arjuna_reflections
    HAS_MODES = True
except Exception:
    HAS_MODES = False
    logger.warning("components.modes not found; using fallback for Arjuna reflections")

# ---------- Utility helpers ----------
def _ensure_list(x):
    """Return x as a list; unwrap (list, meta) tuples; wrap singletons."""
    if x is None:
        return []
    if isinstance(x, tuple) and len(x) >= 1:
        x = x[0]
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def bulletize(items, max_items=3):
    """Flatten one level & coerce to str so join() never fails."""
    out = []
    for itm in _ensure_list(items)[:max_items]:
        if isinstance(itm, (list, tuple)):
            itm = " ".join(map(str, itm))
        else:
            itm = str(itm)
        out.append(f"- {itm}")
    return "\n".join(out) if out else "- (no reflections)"

# ---------- Fallback utilities ----------
def _fallback_embedding(text: str):
    if not PANDAS_NUMPY:
        return None
    if not text or not isinstance(text, str):
        text = "default"
    if HAS_SBERT:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error loading Sentence-BERT model: {e}")
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.random(384)

def _fallback_cosine(a, b):
    if not PANDAS_NUMPY:
        return 0.0
    a = np.asarray(a); b = np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# Public wrappers
def get_embedding(text):
    if HELPERS:
        try:
            return _get_embedding(text)
        except Exception as e:
            logger.warning(f"helper.get_embedding failed, using fallback: {e}")
    return _fallback_embedding(text)

def cosine_similarity(v1, v2):
    if HELPERS:
        try:
            return _cosine_similarity(v1, v2)
        except Exception as e:
            logger.warning(f"helper.cosine_similarity failed, using fallback: {e}")
    return _fallback_cosine(v1, v2)

def mirror_reflections(user_input, df_matrix):
    if HAS_MIRROR:
        try:
            hints = _mirror_reflections(user_input, df_matrix)
            return _ensure_list(hints), None
        except Exception as e:
            logger.warning(f"mirror reflections failed, using fallback: {e}")
    return [
        "Reflect with honesty: what is your true intention?",
        "Consider long-term outcomes over short-term gains.",
        "Choose the path that preserves dignity—yours and others'."
    ], None

def arjuna_reflections(user_input, df_matrix):
    if HAS_MODES:
        try:
            hints = _arjuna_reflections(user_input, df_matrix)
            return _ensure_list(hints)
        except Exception as e:
            logger.warning(f"arjuna reflections failed, using fallback: {e}")
    return [
        "Your reluctance signals attachment—acknowledge it without judgment.",
        "Duty feels heavy when desire leads; align action with higher purpose.",
        "Courage is clarity in motion; take one dharmic step now."
    ]

# ---------- Signals & dynamic action plan ----------
def extract_signals(text: str) -> dict:
    t = (text or "").lower()
    return {
        "urgency": any(k in t for k in ["urgent","deadline","now","asap","today","tonight"]),
        "uncertainty": any(k in t for k in ["uncertain","unknown","ambiguous","confused","unsure"]),
        "stakeholder_conflict": any(k in t for k in ["team","manager","customer","family","partner","board","stakeholder","investor"]),
        "risk_words": any(k in t for k in ["risk","harm","unsafe","privacy","bias","security","safety","breach"]),
        "domain": "work" if any(k in t for k in ["product","ship","release","kpi","roadmap","sprint","contract"]) else \
                  "family" if any(k in t for k in ["family","parent","child","spouse","partner"]) else "general",
    }

# Load YAML templates (with fallback if missing)
ACTION_TEMPLATES = None
try:
    import yaml
    from yaml import SafeLoader
    tmpl_path = Path(project_root) / "action_templates.yaml"
    if tmpl_path.exists():
        ACTION_TEMPLATES = yaml.safe_load(tmpl_path.read_text(encoding="utf-8"))
    else:
        ACTION_TEMPLATES = {}  # fallback
except Exception as e:
    logger.warning(f"YAML not available or failed to load: {e}")
    ACTION_TEMPLATES = {}

def _tmpl_for_tag(tag: str) -> dict:
    tag = (tag or "").lower()
    def get(name): return ACTION_TEMPLATES.get(name, {})
    if "detach" in tag or "karma" in tag:
        return get("detachment")
    if "duty" in tag or "role" in tag:
        return get("duty")
    if "compassion" in tag or "ahimsa" in tag:
        return get("compassion")
    if "truth" in tag or "satya" in tag:
        return get("truth")
    if "equal" in tag or "sama" in tag:
        return get("equality")
    if "self" in tag or "control" in tag or "discipline" in tag:
        return get("self-control")
    if "impermanence" in tag or "time" in tag or "entropy" in tag:
        return get("impermanence")
    return get("detachment") or get("duty") or {}

def _select_steps(pool, signals, k):
    if not isinstance(pool, list):
        return []
    scored = []
    for item in pool:
        step = item.get("step") if isinstance(item, dict) else str(item)
        tags = [t.lower() for t in (item.get("tags", []) if isinstance(item, dict) else [])]
        s = 1.0
        if signals["urgency"] and any(t in tags for t in ["time","reversibility","minimal-step"]): s += 0.4
        if signals["uncertainty"] and any(t in tags for t in ["experiment","uncertainty","probe"]): s += 0.4
        if signals["stakeholder_conflict"] and any(t in tags for t in ["stakeholders","decision-rights","alignment"]): s += 0.35
        if signals["risk_words"] and any(t in tags for t in ["risk","privacy","bias","security","safety","harm"]): s += 0.45
        scored.append((s, step))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [s for _, s in scored[:max(1, k)]]
    if len(top) > 1:
        import random
        rnd = random.Random(hash("::".join(top)) & 0xFFFFFFFF)
        rest = top[1:]
        rnd.shuffle(rest)
        top = [top[0]] + rest
    return top

def generate_action_plan_dynamic(user_input: str, verse_tag: str | None, sizes=(3,3,3)) -> dict:
    signals = extract_signals(user_input)
    tmpl = _tmpl_for_tag(verse_tag)
    short_pool = tmpl.get("short", [])
    med_pool   = tmpl.get("medium", [])
    long_pool  = tmpl.get("long", [])
    short = _select_steps(short_pool, signals, sizes[0]) if short_pool else []
    med   = _select_steps(med_pool,   signals, sizes[1]) if med_pool   else []
    long  = _select_steps(long_pool,  signals, sizes[2]) if long_pool  else []
    if signals["risk_words"] and not any("harm" in s.lower() or "privacy" in s.lower() for s in short + med):
        short = ["Run a quick harm scan: privacy, safety, bias, security."] + short
    if not (short or med or long):
        short = [
            "Clarify your duty and one non-negotiable boundary.",
            "Take one reversible step aligned with duty.",
            "Write what you will measure (harm + dignity + utility).",
        ]
        med = [
            "Pilot with review checkpoints and a stop rule.",
            "Invite a counter-perspective to reduce bias.",
            "Document trade-offs and communicate them plainly.",
        ]
        long = [
            "Create a recurring review ritual (pre/post decision).",
            "Codify policy so it survives handoffs.",
            "Track harm reduction and fairness, not just outcomes.",
        ]
    return {"short": short[:5], "medium": med[:5], "long": long[:5]}

# ---------- Core response generator ----------
def generate_gita_response(mode, df_matrix, user_input=None, top_k=3):
    """Return (response_markdown, top_verse_row) with robust fallbacks."""
    if not user_input or len(user_input.strip()) < 3:
        return "🛑 Please ask a more complete or meaningful question.", None
    if not PANDAS_NUMPY:
        return "⚠️ Error: Required libraries (pandas, numpy) not installed.", None
    if df_matrix is None or getattr(df_matrix, 'empty', True):
        return "⚠️ Error: Verse data not loaded. Please check the CSV file.", None

    col_text = None
    for candidate in ["Short English Translation", "English", "Verse", "Translation", "Summary"]:
        if candidate in df_matrix.columns:
            col_text = candidate
            break
    if col_text is None:
        return "⚠️ Error: Verse text column not found in matrix.", None

    col_id = None
    for candidate in ["Verse ID", "ID", "Ref", "Key"]:
        if candidate in df_matrix.columns:
            col_id = candidate
            break
    col_map = None
    for candidate in ["Symbolic Conscience Mapping", "Mapping", "Theme", "Tag"]:
        if candidate in df_matrix.columns:
            col_map = candidate
            break

    try:
        if 'embedding' not in df_matrix.columns:
            df_matrix = df_matrix.copy()
            df_matrix['embedding'] = df_matrix[col_text].fillna("default").apply(get_embedding)

        query_emb = get_embedding(user_input)
        if query_emb is None:
            return "⚠️ Error: Embeddings unavailable.", None

        df_matrix['similarity'] = df_matrix['embedding'].apply(lambda e: cosine_similarity(query_emb, e))
        if df_matrix['similarity'].isna().all() or df_matrix['similarity'].max() <= 0:
            return "⚠️ Unable to find a matching verse. Try rephrasing your question.", None

        top = df_matrix.sort_values('similarity', ascending=False).head(top_k)
        top_row = top.iloc[0]
    except Exception as e:
        logger.error(f"Embedding/similarity pipeline failed: {e}")
        return f"⚠️ Error computing similarity: {str(e)}", None

    verse_text = str(top_row[col_text])
    verse_id = str(top_row[col_id]) if col_id else "—"
    verse_tag = str(top_row[col_map]) if col_map else "—"

    header = f"""
**Nearest Verse:** `{verse_id}`  
*{verse_text}*  
_Tag:_ `{verse_tag}`
"""

    if mode == "Krishna":
        body = """
**Krishna's Counsel**  
Act without attachment to outcomes. Let clarity guide action; align with duty over impulse.  
**Why this verse?** Your query semantically matched teachings on detachment and right action.
"""
    elif mode == "Krishna-Explains":
        plan = generate_action_plan_dynamic(user_input, verse_tag)
        body = f"""
**Krishna's Teaching — Explained**  
This verse instructs that *right action* is measured by intent and alignment with duty, not by clinging to outcomes.  
Attachment breeds anxiety and bias; detachment clears the mind to see the dharmic path.

**Action Plan**

**Short Term (today–this week)**  
{chr(10).join(f"- {s}" for s in plan['short'])}

**Medium Term (next 2–6 weeks)**  
{chr(10).join(f"- {m}" for m in plan['medium'])}

**Long Term (quarter and beyond)**  
{chr(10).join(f"- {l}" for l in plan['long'])}
"""
    elif mode == "Arjuna":
        hints = arjuna_reflections(user_input, df_matrix)
        body = "**Arjuna's Reflection**\n" + bulletize(hints, max_items=3)
    elif mode in ["Dharma Mirror", "Mirror"]:
        lines, _ = mirror_reflections(user_input, df_matrix)
        body = "**Dharma Mirror**\n" + bulletize(lines, max_items=3)
    elif mode == "Vyasa":
        body = """
**Vyasa's Narration**  
You are at a fork: describe the forces, duties, and attachments at play.  
This verse frames the context so duty can be seen without distortion.
"""
    elif mode in ["Technical"]:
        cols = [c for c in ['similarity', col_id, col_map, col_text] if c and c in top.columns]
        body = f"""
**Technical Trace**  
Top-{top_k} matches (similarity):  
{top[cols].to_string(index=False)}
"""
    else:
        body = "Choose dharma; preserve dignity and long-term harmony."

    footer = """
---
*Tip:* Refine your question to include people, constraints, and the value you refuse to compromise.
"""
    return header + "\n\n" + body + "\n\n" + footer, top_row

# ---------- UI ----------
if STREAMLIT:
    st.title("🪔 DharmaAI – Minimum Viable Conscience")

    if "Usage Journal" not in st.session_state:
        st.session_state["Usage Journal"] = []

    st.subheader("Ask a question to GitaBot")

    if not ENABLE_GITABOT:
        st.warning("🔒 GitaBot integration is currently **disabled**. Please check back later.")
        st.stop()

    available_modes = [
        "Krishna",
        "Krishna-Explains",
        "Arjuna",
        "Dharma Mirror",
        "Vyasa",
        "Technical",
        "Karmic Entanglement Simulator",
        "Forked Fate Contemplation"
    ]
    mode = st.sidebar.radio("Select Mode", available_modes)

    user_input = st.text_input("Your ethical question or dilemma:", value="")

    # Load verse matrix
    matrix_paths = [
        os.path.join(project_root, "data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"),
        os.path.join(project_root, "app/data/gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv"),
        os.path.join(project_root, "gita_dharmaAI_matrix_verse_1_to_2_50_logic.csv")
    ]
    df_matrix = None
    if PANDAS_NUMPY:
        for path in matrix_paths:
            if os.path.exists(path):
                try:
                    df_matrix = pd.read_csv(path, encoding='utf-8')
                    logger.info(f"Loaded verse matrix from {path}")
                    break
                except UnicodeDecodeError:
                    df_matrix = pd.read_csv(path, encoding='ISO-8859-1')
                    logger.info(f"Loaded verse matrix from {path} with ISO-8859-1 encoding")
                    break
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
    if df_matrix is None:
        st.error("⚠️ Error: Could not load verse matrix CSV file. Please check the file path.")
        st.stop()

    if st.button("🔍 Submit"):
        try:
            response, verse_info = generate_gita_response(mode, df_matrix, user_input)
            st.markdown(
                "<div style='border: 1px solid #ddd; padding: 1.5rem; border-radius: 1rem; background-color: #fafafa;'>",
                unsafe_allow_html=True
            )
            if response.startswith(("⚠️", "❌", "🛑")):
                st.error(response)
            else:
                try:
                    vid = None
                    for c in ["Verse ID", "ID", "Ref", "Key"]:
                        if verse_info is not None and c in verse_info:
                            vid = verse_info[c]; break
                    tag = None
                    for c in ["Symbolic Conscience Mapping", "Mapping", "Theme", "Tag"]:
                        if verse_info is not None and c in verse_info:
                            tag = verse_info[c]; break
                    if vid or tag:
                        st.markdown(
                            f"<small>📘 Verse: <code>{vid if vid else '—'}</code> — <em>{tag if tag else '—'}</em></small>",
                            unsafe_allow_html=True
                        )
                except Exception:
                    pass
                st.markdown(response, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Streamlit UI error: {e}")
            st.error(f"⚠️ Unexpected error: {e}")
