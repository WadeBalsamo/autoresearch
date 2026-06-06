"""Canonical VAAMR / PURER definitions — the single source of truth for label spaces.

These mirror QRA `frameworks/VAAMR_FRAMEWORK.md` (v4.0, 5 stages) and
`frameworks/PURER_FRAMEWORK.md` (5 moves). They are embedded (not parsed from QRA at
import time) so the workshop is self-contained, but ``load_qra_markdown`` can pull the
full, richer definitions from a QRA checkout when one is available (used to ground the
synthetic-data prompts).

⚠️ VAAMR is **5-class**. Do not regress to the obsolete 4-stage "VA-MR".
"""
from __future__ import annotations

import os
import re
from typing import Dict, Optional

# --- VAAMR: participant developmental arc (the classification target) --------------
VAAMR_NUM_CLASSES = 5

VAAMR_STAGES: Dict[int, str] = {
    0: "Vigilance",
    1: "Avoidance",
    2: "Attention Regulation",
    3: "Metacognition",
    4: "Reappraisal",
}

# Concise faithful descriptions + a canonical expression per stage (for synth prompts
# and human-readable run logs). Sourced from VAAMR_FRAMEWORK.md stage definitions.
VAAMR_DESCRIPTIONS: Dict[int, str] = {
    0: ("Pain vigilance / attention dysregulation. Attention is colonised by pain; "
        "reactive, fragmented, catastrophising. Canonical: "
        "\"I can't stop thinking about the pain, it's all I can focus on.\""),
    1: ("Attention regulation applied to experiential avoidance. Attentional skill is "
        "deployed to push pain away rather than investigate it. Canonical: "
        "\"When the pain comes I focus hard on my breathing to push it away.\""),
    2: ("Attention regulation. Stable, sustained, volitional attention that stays WITH "
        "present experience without escaping it. Canonical: "
        "\"I kept bringing my attention back to the sensations, just staying with them.\""),
    3: ("Metacognitive awareness. Reflexive distance — observing one's own mental "
        "processes as events. Canonical: "
        "\"I noticed I was getting anxious about the pain, and I could just watch that anxiety.\""),
    4: ("Pain reappraisal. Transformation of the structure/meaning of pain experience; "
        "pain seen as changing, decomposable, lacking fixed significance. Canonical: "
        "\"When I really look at it, the 'pain' is actually many different feelings.\""),
}

# Stages 2,3,4 are the mindfulness skills; 0,1 are pre-mindfulness. Avoidance (1) is the
# load-bearing barrier the therapy must move participants past.
VAAMR_IS_MINDFULNESS = {0: False, 1: False, 2: True, 3: True, 4: True}

# --- PURER: therapist guided-inquiry moves -----------------------------------------
PURER_NUM_MOVES = 5

PURER_MOVES: Dict[int, str] = {
    0: "Phenomenology",
    1: "Utilization",
    2: "Reframing",
    3: "Education",        # Educate / Expectancy
    4: "Reinforcement",
}

PURER_DESCRIPTIONS: Dict[int, str] = {
    0: "Phenomenology — step-by-step elicitation of the participant's practice experience.",
    1: "Utilization — prompting forward application of a skill to everyday life.",
    2: "Reframing — repositioning the participant's report as a MORE concept.",
    3: "Education/Expectancy — psychoeducation about pain/mindfulness plus expectation-setting.",
    4: "Reinforcement — selective affirmation of an adaptive response or insight.",
}

# direction label space for the cue-block tasks
DIRECTIONS = ("regressed", "stayed", "advanced")
DIRECTION_TO_ID = {d: i for i, d in enumerate(DIRECTIONS)}
PROGRESS_DEADBAND = 0.15  # matches QRA analysis/mechanism.py and mindfulbert_dataset.py


def vaamr_label_map() -> Dict[str, int]:
    """short_name(lower) -> id, tolerant of QRA's lower-cased label strings."""
    return {name.lower(): i for i, name in VAAMR_STAGES.items()}


def vaamr_name(label_id: int) -> str:
    return VAAMR_STAGES.get(int(label_id), f"stage_{label_id}")


def purer_name(move_id) -> Optional[str]:
    if move_id is None:
        return None
    return PURER_MOVES.get(int(move_id))


def framework_block_for_prompt() -> str:
    """A compact framework description block to ground synthetic-data prompts."""
    lines = ["VAAMR — participant developmental stages (0..4):"]
    for i in range(VAAMR_NUM_CLASSES):
        lines.append(f"  {i} {VAAMR_STAGES[i]}: {VAAMR_DESCRIPTIONS[i]}")
    lines.append("")
    lines.append("PURER — therapist guided-inquiry moves (0..4):")
    for i in range(PURER_NUM_MOVES):
        lines.append(f"  {i} {PURER_DESCRIPTIONS[i]}")
    return "\n".join(lines)


def load_qra_markdown(qra_repo: str) -> Optional[str]:
    """Best-effort: return the raw VAAMR + PURER framework markdown from a QRA checkout.

    Used only to enrich synthetic-data prompts with the full definitions/exemplars when a
    QRA repo path is provided. Returns None if the files are not found.
    """
    parts = []
    for fname in ("VAAMR_FRAMEWORK.md", "PURER_FRAMEWORK.md"):
        p = os.path.join(qra_repo, "frameworks", fname)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
            # strip the long PARSER CONTRACT comment block to keep prompts lean
            txt = re.sub(r"<!--.*?-->", "", txt, flags=re.DOTALL)
            parts.append(txt.strip())
    return "\n\n".join(parts) if parts else None
