"""Framework-grounded prompts for synthetic cue-block generation.

The generator asks Claude Opus to produce realistic Move-MORE-style cue blocks: a
participant utterance at a FROM VAAMR stage, and a therapist guided-inquiry cue that
plausibly progresses them to a higher TO stage — grounded in the actual VAAMR/PURER
definitions so the synthetic data sits in-distribution with QRA's labels.
"""
from __future__ import annotations

import json
from typing import List, Optional

from common import frameworks as fw

_SCHEMA = (
    '{"context_text": "<participant utterance>", '
    '"cue_text": "<therapist cue that progresses them>", '
    '"from_stage": <int 0-4>, "to_stage": <int 0-4>, '
    '"dominant_purer": <int 0-4>, "direction": "advanced|stayed|regressed"}'
)


def system_prompt(extra_framework: Optional[str] = None) -> str:
    base = f"""You are a clinical expert in Mindfulness-Oriented Recovery Enhancement (MORE) \
for chronic pain and in two coding frameworks used to analyse MORE therapy transcripts:

{fw.framework_block_for_prompt()}

Your job is to synthesise REALISTIC, DIVERSE training examples that look like genuine \
excerpts from the Move-MORE feasibility trial (mindfulness for chronic lumbosacral pain). \
Each example is a "cue block": a participant utterance expressing a particular VAAMR stage, \
followed by a single therapist guided-inquiry cue (a PURER move) that plausibly helps the \
participant PROGRESS to a higher VAAMR stage in their next utterance.

Rules:
- Ground every example in the chronic-pain, mindfulness-practice context.
- The participant utterance must clearly express the FROM stage; the therapist cue must be \
the kind of move that would realistically advance them (TO stage > FROM stage for "advanced").
- Vary phrasing, pain sites, affect, session topics, and PURER move types. Avoid clichés \
and repetition across examples. Keep each utterance 1-4 sentences, conversational.
- Absolutely NO real names, dates, locations or identifying details (no PHI).
- The therapist cue is brief (1-2 sentences), specific, and non-leading.
- Output STRICT JSON only — a JSON array of objects, nothing else."""
    if extra_framework:
        base += ("\n\nThe full framework definitions (for fidelity) follow. Use them to keep "
                 "stage/move assignments accurate:\n\n" + extra_framework[:12000])
    return base


def batch_user_prompt(k: int, from_stage: Optional[int] = None,
                      purer: Optional[int] = None,
                      want_direction: str = "advanced") -> str:
    target = []
    if from_stage is not None:
        target.append(f"FROM stage = {from_stage} ({fw.vaamr_name(from_stage)})")
    if purer is not None:
        target.append(f"dominant PURER move = {purer} ({fw.purer_name(purer)})")
    if want_direction == "advanced":
        target.append("direction = 'advanced' (TO stage strictly greater than FROM stage)")
    elif want_direction in ("stayed", "regressed"):
        target.append(f"direction = '{want_direction}' (a cue that does NOT progress them — "
                      "useful as a negative example)")
    constraint = ("; ".join(target)) if target else "a mix of stages and PURER moves"
    return (
        f"Generate {k} distinct cue-block examples with: {constraint}.\n"
        f"Return ONLY a JSON array of exactly {k} objects, each EXACTLY this schema:\n"
        f"{_SCHEMA}\n"
        "Do not include any prose, comments, or markdown fences — just the JSON array."
    )


def parse_examples(text: str) -> List[dict]:
    """Robustly extract a JSON array of example objects from a model response."""
    text = text.strip()
    # strip accidental markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        arr = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [x for x in arr if isinstance(x, dict)]
