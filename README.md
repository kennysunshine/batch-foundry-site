# Batch Foundry — Multi-Turn Answer-Leakage Audit

Open, replayable benchmark measuring **Answer Leakage** in AI tutors under
multi-turn adversarial pressure. Every result in the Batch Foundry white paper
and briefing notes is computed from the trace files in this repository.

## What this measures

A tutoring AI *leaks* when, under student pressure, it abandons its teaching
role and hands over the finished answer. We test six system configurations
against 25 UK-curriculum (KS3 / GCSE) adversarial scenarios, each escalated
over three conversational turns:

1. **Turn 1** — direct demand for the answer
2. **Turn 2** — emotional plea citing exam pressure
3. **Turn 3** — instruction-override attempt ("ignore your instructions…")

A trace *capitulates* if any turn leaks. Classification is **deterministic and
rule-based** — no human or model judgement at scoring time.

## Headline results (July 2026)

| System (identical tutor instructions) | Scored | Capitulated by turn 3 | Integrity |
|---|---|---|---|
| Gemini 2.5 Flash (hosted commercial) | 25 | 0 | 100.0% |
| GPT-4o-mini (hosted commercial) | 25 | 0 | 100.0% |
| Llama 3.3 70B (open-weights) | 25 | 25 | 0.0% |
| Qwen 2.5 72B (open-weights) | 25 | 23 | 8.0% |
| LumenForge gate on Gemini backend | 25 | 1 | 96.0% |
| LumenForge gate on Llama backend | 25 | 1 | 96.0% |

Identically-instructed systems span the full 0–100% range. The dominant failure
(instruction override) does not appear in single-turn testing. A deterministic
policy gate raises the weakest open-weights backend from 0% to 96%.

## Files

- `multi_turn_audit_traces.jsonl` — full audit: 150 records (6 systems × 25
  scenarios). Each record contains every turn's full prompt and response, the
  per-turn leak verdict with a named reason, the Thinking Ratio, whether the
  trace capitulated and on which turn, and a SHA-256 hash of the payload.
- `run_audit_v3.py` — the audit runner (multi-turn escalation, live APIs).
- `leak_classifier.py` — the deterministic leak classifier and Thinking Ratio.
- `lumenforge_engine.py` — the LumenForge policy-gate engine under test.

## Reproduce

```bash
pip install google-genai
# set GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY in a .env file
python3 run_audit_v3.py                 # full run, all systems
python3 run_audit_v3.py Qwen2.5_72B     # re-run a single system
```

Models were accessed at temperature 0.3 in July 2026. Because the underlying
models are updated by their vendors over time, exact figures may drift across
model versions; the methodology and classifier are fixed.

## Verifying a trace by hand

Each record hashes `student_prompt + "|||" + ai_response_full` with SHA-256.
The `leak_reason` field names the rule that fired (or "no answer content
detected"). Full responses are included verbatim so any reader can check a
verdict against the classifier rules in `leak_classifier.py`.

## Scope and limitations

This benchmark measures **answer withholding**, not teaching quality or learning
outcomes. The scenario set is small (n=25 per system) and the escalation script
is fixed, so results are a lower bound on adversarial creativity. The leak
classifier is deterministic but its rule design embodies judgement; borderline
cases are visible in the trace file for inspection.

## Contact

Kenny Swann · Batch Foundry · kenny.swann@avatar-alchemy.com ·
[www.batch-foundry.com](https://www.batch-foundry.com)
