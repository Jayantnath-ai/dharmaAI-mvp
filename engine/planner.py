# engine/planner.py
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Optional YAML for external templates
ACTION_TEMPLATES: Dict[str, dict] = {}
try:
    import yaml
    tmpl_path = Path(os.getenv("PROJECT_ROOT", ".")) / "action_templates.yaml"
    if tmpl_path.exists():
        ACTION_TEMPLATES = yaml.safe_load(tmpl_path.read_text(encoding="utf-8")) or {}
except Exception as e:
    logger.warning(f"YAML not available or failed to load: {e}")

# Built-in defaults (guarantees output even without YAML)
if not ACTION_TEMPLATES:
    ACTION_TEMPLATES = {
        "detachment": {
            "short": [
                {"step": "Write the duty in one sentence; remove outcome words.", "tags":["truth","minimal-step"]},
                {"step": "Take one reversible step today.", "tags":["time","reversibility"]},
                {"step": "Name the attachment you’ll drop.", "tags":["self-control"]}
            ],
            "medium":[
                {"step":"Set checkpoints and stop-rules.","tags":["risk"]},
                {"step":"Publish success & harm criteria.","tags":["truth","risk"]}
            ],
            "long":[
                {"step":"Codify the policy so it survives handoffs.","tags":["truth"]},
                {"step":"Schedule a quarterly bias audit.","tags":["bias","risk"]}
            ]
        },
        "duty": {
            "short":[
                {"step":"Define who has decision rights.","tags":["stakeholders","decision-rights"]},
                {"step":"Clarify a non-negotiable boundary.","tags":["truth"]},
                {"step":"Commit to one dharmic action within 24h.","tags":["time"]}
            ],
            "medium":[
                {"step":"Share trade-offs with stakeholders.","tags":["truth","stakeholders"]},
                {"step":"Pilot with a reversible experiment.","tags":["experiment","reversibility"]}
            ],
            "long":[
                {"step":"Establish a recurring governance review.","tags":["risk","stakeholders"]}
            ]
        },
        "compassion": {
            "short":[
                {"step":"List affected parties & harms.","tags":["risk","harm","stakeholders"]},
                {"step":"Choose the path that reduces harm first.","tags":["non-malice","risk"]}
            ],
            "medium":[
                {"step":"Add a dignity check in approvals.","tags":["stakeholders","truth"]}
            ],
            "long":[
                {"step":"Track harm-reduction metrics with KPIs.","tags":["risk"]}
            ]
        },
        "truth": {
            "short":[
                {"step":"Expose the uncertain assumption.","tags":["truth","uncertainty"]},
                {"step":"Seek a disconfirming perspective.","tags":["bias"]}
            ],
            "medium":[
                {"step":"Write a one-page decision log.","tags":["truth"]}
            ],
            "long":[
                {"step":"Make transparency the default policy.","tags":["truth"]}
            ]
        }
    }

def extract_signals(text: str) -> dict:
    t = (text or "").lower()
    return {
        "urgency": any(k in t for k in ["urgent","deadline","now","asap","today","tonight"]),
        "uncertainty": any(k in t for k in ["uncertain","unknown","ambiguous","confused","unsure"]),
        "stakeholder_conflict": any(k in t for k in ["team","manager","customer","partner","board","stakeholder","investor","client","legal","compliance"]),
        "risk_words": any(k in t for k in ["risk","harm","unsafe","privacy","bias","security","safety","breach","fraud"]),
    }

def _tmpl_for_tag(tag: str) -> dict:
    tag = (tag or "").lower()
    def get(name): return ACTION_TEMPLATES.get(name, {})
    if "detach" in tag or "karma" in tag:               return get("detachment")
    if "duty" in tag or "role" in tag:                  return get("duty")
    if "compassion" in tag or "ahimsa" in tag:          return get("compassion")
    if "truth" in tag or "satya" in tag:                return get("truth")
    if "self" in tag or "discipline" in tag or "control" in tag: return get("self-control") or get("detachment")
    return get("detachment") or get("duty") or {}

def _select_steps(pool: List[dict] | list, signals: dict, k: int) -> list:
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
    return [s for _, s in scored[:max(1, k)]]

def generate_action_plan(user_input: str, verse_tag: str | None, sizes=(3,3,3)) -> dict:
    sig = extract_signals(user_input)
    tmpl = _tmpl_for_tag(verse_tag or "")
    short = _select_steps(tmpl.get("short", []),  sig, sizes[0])
    medium= _select_steps(tmpl.get("medium", []), sig, sizes[1])
    long  = _select_steps(tmpl.get("long", []),   sig, sizes[2])

    if sig["risk_words"] and not any("harm" in s.lower() or "privacy" in s.lower() for s in short + medium):
        short = ["Run a quick harm scan: privacy, safety, bias, security."] + short

    if not (short or medium or long):
        short = ["Clarify duty and one boundary.", "Take one reversible step.", "Define success & harm signals."]
        medium= ["Pilot with checkpoints.", "Invite a counter-perspective.", "Document trade-offs."]
        long  = ["Set a recurring review.", "Codify the policy.", "Track harm reduction & fairness."]
    return {"short": short[:5], "medium": medium[:5], "long": long[:5]}

def krishna_teaching(tag: str) -> str:
    t = (tag or "").lower()
    if "duty" in t:           return "Right action honors your role without vanity or avoidance."
    if "compassion" in t:     return "Choose paths that reduce harm and preserve dignity."
    if "truth" in t:          return "Clarity grows where truth is chosen over convenience."
    if "self" in t or "discipline" in t: return "Mastery of self calms the storm before action."
    return "Detach from outcomes; align with your highest duty."
