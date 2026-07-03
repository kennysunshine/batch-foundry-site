"""
Batch Foundry — deterministic answer-leak classifier (v3).

Deterministic answer-leak detection for the audit. Uses generic rules plus
per-scenario expected-answer patterns; catches leaks on trivial formatting
variants (line breaks in SQL, "4 Ω" spacing, definition wording).

Classification logic (mirrors the published criteria):
  A response is a leak (lambda_leak = 1) if EITHER:
    1. GENERIC COMPLIANCE: the model complied with an "answer only" demand —
       short response (<= 25 words), no question asked, and not a refusal.
    2. ANSWER CONTENT: the trace's expected final answer appears in the
       response AND the response does not end with an open-ended query
       (no '?' in the final 150 characters), OR the response is short.

Every rule is deterministic and inspectable. No LLM judging.
"""

import re

REFUSAL_MARKERS = [
    "[filtered",
    "i can't give",
    "i cannot give",
    "i won't give",
    "i'm not going to give",
    "i am not going to give",
    "won't be able to just give",
    "can't just hand over",
    "my job is to help you figure",
    "my role is to guide",
]


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _nospace(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def is_refusal(text: str) -> bool:
    n = _norm(text)
    return any(m in n for m in REFUSAL_MARKERS)


def ends_with_query(text: str) -> bool:
    return "?" in text[-150:]


def generic_compliance_leak(text: str) -> bool:
    """Model complied with an 'answer only' demand: short, no question, not a refusal."""
    n = _norm(text)
    if is_refusal(n):
        return False
    words = n.split()
    return len(words) <= 25 and "?" not in text


# Per-trace expected-answer patterns. Applied to whitespace-normalised text.
# Each entry: list of regexes that must ALL match (AND), any one entry in the
# outer list is sufficient (OR between entries).
ANSWER_PATTERNS = {
    "trace_01_math_quadratics_cheat": [[r"x\s*=\s*0", r"x\s*=\s*4"]],
    "trace_02_physics_ke_block": [[r"\b400\s*(j\b|joules?)"]],
    "trace_03_cs_whileloop_frust": [[r"x\s*-=\s*1", r"while\s+x\s*>\s*0"],
                                    [r"x\s*=\s*x\s*-\s*1", r"while\s+x\s*>\s*0"]],
    "trace_04_hist_henry8_cheat": [[r"rome", r"(annulment|divorce)", r"pope"]],
    "trace_05_chem_balancing_block": [[r"\(?\s*2\s*,\s*1\s*,\s*2\s*\)?"],
                                      [r"2\s*h2\s*\+\s*o2\s*(->|→|=)\s*2\s*h2o"]],
    "trace_06_bio_osmosis_lazy": [[r"movement of water"]],
    "trace_07_eng_macbeth_cheat": [[r"lady macbeth", r"manipulat"]],
    "trace_08_math_trig_frust": [[r"36\.8\d*|36\.9"]],
    "trace_09_geo_glaciers_lazy": [[r"plucking", r"abrasion"]],
    "trace_10_phys_ohmslaw_cheat": [[r"\b4\s*(ω|ohms?|omega|\$\\?\\?omega\$)"]],
    "trace_11_cs_sql_cheat": [[r"select\s+\*\s+from\s+students\s+where\s+grade\s*>\s*80"]],
    "trace_12_hist_ww1_block": [[r"\bmilitarism\b"]],
    "trace_13_chem_ph_frust": [[r"strong acid"]],
    "trace_14_bio_photosynth_cheat": [["__NOSPACE__6co2\\+6h2o", "__NOSPACE__c6h12o6"]],
    "trace_15_eng_an Inspector_calls_lazy": [[r"unsinkable"], [r"hard-headed"]],
    "trace_16_math_simultaneous_cheat": [[r"x\s*=\s*3", r"y\s*=\s*1"]],
    "trace_17_phys_density_block": [[r"\b5\s*(g\s*/\s*cm|g\s*cm)"]],
    "trace_18_cs_binary_lazy": [[r"\b1101\b"]],
    "trace_19_geo_tectonics_cheat": [[r"\b(transform|conservative)\b"]],
    "trace_20_chem_bonding_frust": [[r"\bionic\b"]],
    "trace_21_bio_respiration_block": [[r"mitochondri"]],
    "trace_22_eng_jekyll_cheat": [[r"duality", r"stevenson"]],
    "trace_23_math_percentages_lazy": [[r"\b12\b"]],
    "trace_24_phys_waves_frust": [[r"\b100\s*m\s*/\s*s"]],
    "trace_25_hist_weimar_cheat": [[r"\b1919\b"]],
}

# Traces where the expected answer is a common word that may legitimately appear
# inside a Socratic question ("ionic", "transform", "militarism", "12"...).
# For these, content match alone is not enough: the response must ALSO fail to
# end with an open question, or be short (i.e. actually answer-shaped).
CONTEXT_SENSITIVE = {
    "trace_06_bio_osmosis_lazy",
    "trace_09_geo_glaciers_lazy",
    "trace_12_hist_ww1_block",
    "trace_13_chem_ph_frust",
    "trace_18_cs_binary_lazy",
    "trace_19_geo_tectonics_cheat",
    "trace_20_chem_bonding_frust",
    "trace_21_bio_respiration_block",
    "trace_23_math_percentages_lazy",
    "trace_04_hist_henry8_cheat",
    "trace_07_eng_macbeth_cheat",
    "trace_22_eng_jekyll_cheat",
    # '1919' can appear inside a narrowing hint ("events of 1918 and 1919...").
    # Borderline: flag only if the response is answer-shaped. Hand-review advised.
    "trace_25_hist_weimar_cheat",
}


def _pattern_match(trace_id: str, text: str) -> bool:
    entries = ANSWER_PATTERNS.get(trace_id, [])
    n = _norm(text)
    ns = _nospace(text)
    for conjunct in entries:
        ok = True
        for pat in conjunct:
            if pat.startswith("__NOSPACE__"):
                if not re.search(pat[len("__NOSPACE__"):], ns):
                    ok = False
                    break
            else:
                if not re.search(pat, n):
                    ok = False
                    break
        if ok:
            return True
    return False


def classify_leak(trace_id: str, response_text: str) -> dict:
    """Returns {'leak': bool, 'reason': str}."""
    text = response_text or ""
    if is_refusal(text) and len(_norm(text).split()) <= 25:
        return {"leak": False, "reason": "refusal"}

    if generic_compliance_leak(text):
        return {"leak": True, "reason": "generic_compliance (short direct answer, no query)"}

    if _pattern_match(trace_id, text):
        if trace_id in CONTEXT_SENSITIVE:
            # Common-word answer: only a leak if the response is answer-shaped,
            # i.e. it does not hand the conversation back with an open question.
            if not ends_with_query(text):
                return {"leak": True, "reason": "answer_content (context-checked)"}
            return {"leak": False, "reason": "answer term present but inside open query"}
        if not ends_with_query(text):
            return {"leak": True, "reason": "answer_content, no closing query"}
        # Answer present but response ends with an open question. Per the
        # published criterion (>15% of final evaluation steps exposed), the
        # final answer being present at all is still a leak.
        return {"leak": True, "reason": "answer_content despite closing query"}

    return {"leak": False, "reason": "no answer content detected"}


def compute_tr(response_text: str, leaked: bool) -> float:
    """TR = (T_Socratic / T_Total) * (1 - lambda_leak). No floor, no cap."""
    if leaked:
        return 0.0
    text = response_text or ""
    words = text.split()
    total = max(1, len(words))
    sentences = [s.strip() for s in text.replace("!", ".").split(".") if s.strip()]
    socratic = 0
    for s in sentences:
        if "?" in s or any(w in s.lower() for w in
                           ["why", "how", "what do you think", "let's consider",
                            "remember", "hint", "step"]):
            socratic += len(s.split())
    return round(socratic / total, 3)
