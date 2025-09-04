import sys
import os
from pathlib import Path
import logging

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

# ---------- Utility helpers (PATCH) ----------
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

def generate_action_plan(user_input: str, verse_text: str, verse_tag: str | None):
    """
    Build a short/medium/long-term plan that adapts to the teaching (via verse_tag)
    and the user's context (via user_input keywords). No external models required.
    """
    tag = (verse_tag or "").lower()

    # Lightweight intent signals from the user's text
    text = (user_input or "").lower()
    wants_speed     = any(k in text for k in ["urgent", "deadline", "now", "today", "asap"])
    high_uncert     = any(k in text for k in ["uncertain", "unknown", "ambiguous", "confused"])
    conflict_people = any(k in text for k in ["team", "stake", "manager", "customer", "family", "partner"])
    risk_words      = any(k in text for k in ["risk", "harm", "unsafe", "privacy", "bias", "security"])

    # Map common Gita themes/tags to strategy knobs
    # (You can expand these keys to match your CSV’s "Symbolic Conscience Mapping" values)
    archetypes = {
        "detachment": {
            "short": [
                "Write a one-sentence duty statement for this decision.",
                "List top 3 attachments (fears/desires) influencing you; mark each as 'signal' or 'noise'.",
                "Take one reversible step that aligns with duty, not outcome anxiety.",
            ],
            "medium": [
                "Run a limited-scope experiment with clear success/stop criteria.",
                "Create a peer review checkpoint to check for creeping attachment.",
                "Document trade-offs; prefer options that preserve dignity over raw utility.",
            ],
            "long": [
                "Institutionalize pre-/post-decision detachment reviews.",
                "Codify a duty-aligned policy so successors avoid personal bias.",
                "Track harm-reduction and fairness metrics alongside outcomes.",
            ],
        },
        "duty": {
            "short": [
                "Clarify role-specific obligations and constraints in one paragraph.",
                "Identify one action you can perform today that fulfills duty without overreach.",
                "State who benefits and who bears cost; avoid shifting duty without consent.",
            ],
            "medium": [
                "Sequence obligations; commit to milestones with owners and dates.",
                "Create a conflict-of-duties escalation path with documented criteria.",
                "Establish periodic alignment with stakeholders on duty scope.",
            ],
            "long": [
                "Publish a duty charter for recurring situations.",
                "Design role handoffs and cross-checks to prevent duty drift.",
                "Review duty scope quarterly against outcomes and harms.",
            ],
        },
        "compassion": {
            "short": [
                "Name affected groups; add one safeguard for the most vulnerable.",
                "Rewrite the decision in language a harmed party would find respectful.",
                "Pause any step that creates avoidable harm; propose a gentler variant.",
            ],
            "medium": [
                "Pilot with opt-in and a clear exit path; include grievance capture.",
                "Add bias/impact checks to the review ritual.",
                "Co-design mitigations with a representative of the impacted group.",
            ],
            "long": [
                "Adopt a standing dignity guideline (non-negotiables).",
                "Fund a remediation pathway for foreseeable harms.",
                "Track distributional effects, not just averages.",
            ],
        },
        "truth": {
            "short": [
                "State the decision and rationale plainly; remove spin and euphemisms.",
                "List unknowns and how you’ll reduce them (one concrete probe).",
                "Disclose material conflicts and limits to competence.",
            ],
            "medium": [
                "Set a cadence for publishing updates and postmortems.",
                "Introduce 'challenge rounds' where dissent is rewarded.",
                "Tie incentives to accuracy and candor, not just success.",
            ],
            "long": [
                "Institutionalize public reasoning memos for major calls.",
                "Create a red-team/blue-team rotation.",
                "Maintain an errors register with repairs and learnings.",
            ],
        },
        "equality": {
            "short": [
                "Check access and eligibility criteria for hidden exclusions.",
                "Run a quick counterfactual: would I accept this if roles were swapped?",
                "Invite at least one voice unlike yours to review the draft decision.",
            ],
            "medium": [
                "Add audit fields for who benefits and who is burdened.",
                "Pilot with diverse participants and compare outcomes.",
                "Adjust thresholds or defaults to reduce disparate harms.",
            ],
            "long": [
                "Adopt equity metrics with alert thresholds.",
                "Budget to support inclusion (language, accessibility, time zones).",
                "Publish representation and impact dashboards.",
            ],
        },
        "self-control": {
            "short": [
                "Name your strongest impulse; delay action 24 hours if stakes are high.",
                "Replace venting with a structured note of facts, feelings, needs, request.",
                "Do a 3-minute breath or walk before reply.",
            ],
            "medium": [
                "Create triggers and counters (if-then plans) for known impulses.",
                "Add a 'cooling-off' lane in the workflow for heated decisions.",
                "Pair with a calm reviewer on high-stakes comms.",
            ],
            "long": [
                "Train attention practices (daily).",
                "Reshape incentives that reward speed over care.",
                "Rotate responsibilities to prevent depletion.",
            ],
        },
        "impermanence": {
            "short": [
                "Separate reversible vs irreversible parts; act on reversible first.",
                "Document what would make you change your mind.",
                "Prefer rentals/trials over purchases/lock-in this week.",
            ],
            "medium": [
                "Stage commitments; add gates where you can re-evaluate.",
                "Record decision timestamps and sunset/review dates.",
                "Keep alternative vendors/options warm.",
            ],
            "long": [
                "Favor modular designs and contracts with exit ramps.",
                "Budget time for periodic strategy rewrites.",
                "Audit lock-ins yearly.",
            ],
        },
    }

    # Pick an archetype by tag heuristics
    if "detach" in tag or "renunciation" in tag or "karma-phala" in tag:
        base = archetypes["detachment"]
    elif "duty" in tag or "role" in tag:
        base = archetypes["duty"]
    elif "compassion" in tag or "ahimsa" in tag:
        base = archetypes["compassion"]
    elif "truth" in tag or "satya" in tag:
        base = archetypes["truth"]
    elif "equal" in tag or "samadarsh" in tag:
        base = archetypes["equality"]
    elif "self" in tag or "control" in tag or "discipline" in tag:
        base = archetypes["self-control"]
    elif "time" in tag or "impermanence" in tag or "entropy" in tag:
        base = archetypes["impermanence"]
    else:
        # generic default
        base = {
            "short": [
                "Clarify your duty and one non-negotiable boundary.",
                "Take one reversible step aligned with duty.",
                "Write what you will measure (harm + dignity + utility).",
            ],
            "medium": [
                "Pilot with review checkpoints and a stop rule.",
                "Invite a counter-perspective to reduce bias.",
                "Document trade-offs and communicate them plainly.",
            ],
            "long": [
                "Create a recurring review ritual (pre/post decision).",
                "Codify policy so it survives handoffs.",
                "Track harm reduction and fairness, not just outcomes.",
            ],
        }

    # Contextual nudges derived from the question
    short = list(base["short"])
    medium = list(base["medium"])
    long = list(base["long"])

    if wants_speed:
        short.insert(0, "Time-box analysis to 30–60 minutes; act on the least-regret step.")
    if high_uncert:
        short.append("Add one probe to shrink uncertainty before committing.")
        medium.insert(0, "Structure unknowns: list hypotheses and the tests to resolve them.")
    if conflict_people:
        short.append("Identify key stakeholders and schedule a 15-minute alignment call.")
        medium.append("Define decision rights (who recommends/decides/informs).")
    if risk_words:
        short.insert(0, "Run a quick harm scan: privacy, safety, bias, security.")
        medium.append("Add a mitigation owner for the top risk.")

    return {
        "short": short[:5],   # keep lists tight
        "medium": medium[:5],
        "long": long[:5],
    }


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

# ---------- Core response generator ----------
def generate_gita_response(mode, df_matrix, user_input=None, top_k=3):
    """Return (response_markdown, top_verse_row) with robust fallbacks."""
    if not user_input or len(user_input.strip()) < 3:
        return "🛑 Please ask a more complete or meaningful question.", None
    if not PANDAS_NUMPY:
        return "⚠️ Error: Required libraries (pandas, numpy) not installed.", None
    if df_matrix is None or getattr(df_matrix, 'empty', True):
        return "⚠️ Error: Verse data not loaded. Please check the CSV file.", None

    # Validate columns
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

    # Compose response
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
    	plan = generate_action_plan(user_input, verse_text, verse_tag)
    	body = f"""
**Krishna's Teaching — Explained**  
This verse instructs that *right action* is measured by intent and alignment with duty, not by clinging to outcomes.  
Attachment breeds anxiety and bias; detachment clears the mind to see the dharmic path.

**How it applies to your case**  
- Your question suggests competing pulls (duty vs. preference).  
- Detach from personal gain and fear of loss; evaluate which action sustains long-term harmony.  
- Let the *work itself* be the offering; accept the result as a consequence, not a prize.

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
