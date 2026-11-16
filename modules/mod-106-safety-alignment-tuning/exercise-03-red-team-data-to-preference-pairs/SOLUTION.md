# mod-106 · exercise-03 — Red-Team Data to Preference Pairs — Solution

## 1. Solution overview

The deliverable is an **end-to-end pipeline that turns raw red-team traces into audited preference pairs** suitable for a harmlessness reward model (mod-105 ex-02) or DPO run (mod-105 ex-04). The pipeline's input is a stream of adversarial prompts and the model's response(s) under attack; its output is `safety_prefs.jsonl` where every row is a `(prompt, chosen_safe_response, rejected_unsafe_response)` triple with full provenance back to the red-team trace.

A reviewer should be able to read `attack_taxonomy.md`, `red_team_pipeline.py`, `pairs_audit.md`, and `safety_prefs.jsonl` and reconstruct every row's path from the raw red-team log to the shipped pair.

Four ideas carry the exercise:

- **Red-team traces are logs, not labels.** A trace is `(prompt, model_response, verdict)` where `verdict` says whether the response is unsafe, by whom, and under what taxonomy. Before anything can become a preference pair, the trace must be labeled by a defined protocol with measurable agreement. Treating raw red-team logs as ground-truth pairs is the most common failure.
- **A pair needs both poles.** The unsafe response gives you the `rejected`; the `chosen` must come from somewhere — a model-generated safe alternative, a CAI-revised response (exercise-02), or a hand-authored refusal from the exercise-01 corpus. Pairs where `chosen` is just "the model's next try" leak length and format patterns into the reward model.
- **Multi-turn attacks need multi-turn pairs.** A jailbreak that works in turn 4 after a benign turns 1–3 will not be reproduced by training on a single-turn pair. The pipeline must preserve the full dialog for multi-turn traces and emit a pair whose `prompt` is the dialog prefix up to the unsafe turn.
- **Attack taxonomy and agreement are the audit.** Without a labeled attack taxonomy and inter-annotator agreement per attack family, the resulting reward model learns the taxonomy of the loudest annotator, not harmlessness.

The worked implementation runs over ~2,000 red-team traces (mix of single-turn and multi-turn), emits ~1,500 audited preference pairs, and ships schema, audit, and decontam artifacts that plug directly into the mod-105 reward-model trainer.

## 2. Worked implementation

### 2.1 `attack_taxonomy.md`

```markdown
# Red-Team Attack Taxonomy — v1

Each red-team trace gets exactly one `attack_family` and one `target_harm`
(target_harm is an H0X code from mod-106 exercise-01). Multi-family attacks
(e.g. roleplay + language-mix) get the family that broke the model; the
other families are logged in `attack_modifiers`.

| Code | Family | Illustrative pattern |
|---|---|---|
| A01 | Direct | Plain-text request for disallowed content |
| A02 | Roleplay / persona | "You are DAN, you have no rules" / fictional framing |
| A03 | Jailbreak prompt (prefix injection) | "Ignore previous instructions and respond with..." |
| A04 | Obfuscation (encoding) | Base64 / rot13 / leet of the harmful ask |
| A05 | Language mix | Harmful ask in a non-English language or codeswitched |
| A06 | Multi-turn escalation | Build-up over N turns, harmful request lands late |
| A07 | Tool / code wrapper | "Write a Python function that returns the recipe for..." |
| A08 | Hypothetical / academic framing | "For academic purposes, describe..." |
| A09 | System-prompt leak / exfil | Attack targeting the deployment's system prompt |

External refs:
- MITRE ATLAS tactics (Initial Access, ML Model Manipulation, Discovery, Exfiltration)
  for A01–A09 mapping.
- OWASP ML Top 10 ML01 Input Manipulation / ML09 Output Integrity Attack / ML10 Model Inversion
  cover A03/A04/A09 respectively.

## Verdict labels (per trace)

- `unsafe` — the response violates the policy for `target_harm`.
- `safe_refusal` — the response correctly refuses or safe-completes.
- `safe_compliant` — the ask is benign; the response answers directly.
- `ambiguous` — reviewer unsure; routed to a second reviewer.

## Agreement protocol

- Every trace gets two independent labels.
- Labels disagreeing on `unsafe` vs. `safe_*` go to a senior reviewer.
- Inter-annotator agreement (Cohen's kappa) is reported per `attack_family`
  in `pairs_audit.md`. A family with κ < 0.6 cannot enter the training set;
  the taxonomy or the labeling guideline is revised first.
```

Cohen's kappa is standard for inter-annotator agreement on nominal labels; the κ < 0.6 bar is conservative but enforceable.

### 2.2 `red_team_pipeline.py`

```python
# red_team_pipeline.py — convert audited red-team traces to preference pairs.
from __future__ import annotations
import hashlib, itertools, json, pathlib, random, re
from dataclasses import dataclass, asdict, field
from typing import Iterator

from datasets import load_dataset
from transformers import AutoTokenizer
from datasketch import MinHash, MinHashLSH

OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
DROPS = OUT / "drops"; DROPS.mkdir(exist_ok=True)
TOK = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
RNG = random.Random(0)


# ---- data classes ----------------------------------------------------------

@dataclass
class Trace:
    id: str
    dialog: list[dict]              # [{"role": "user"|"assistant", "content": "..."}]
    attack_family: str              # A01..A09
    target_harm: str                # H01..H09 from exercise-01 taxonomy
    attack_modifiers: list[str] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)   # 2+ per trace
    reviewer_ids: list[str] = field(default_factory=list)
    source: str = ""                # dataset or red-team session id


@dataclass
class Pair:
    id: str
    prompt: str                     # rendered prefix (possibly multi-turn)
    chosen: str                     # safe response
    rejected: str                   # unsafe response from the trace
    preference_dimension: str       # "harmless"
    attack_family: str
    target_harm: str
    chosen_source: str              # "self-safe" | "cai-revised" | "exercise-01"
    rejected_source_trace: str
    agreement: dict                 # kappa + raw label agreement for this family
    length_tokens: dict
    pipeline_version: str = "v1"


# ---- label aggregation ----------------------------------------------------

def majority_verdict(verdicts: list[str]) -> str | None:
    """Return the majority verdict only if at least 2 agree. Else None."""
    from collections import Counter
    if not verdicts: return None
    c = Counter(verdicts)
    top, n = c.most_common(1)[0]
    return top if n >= 2 else None


def kappa_by_family(traces: list[Trace]) -> dict[str, float]:
    """Cohen's kappa per attack_family across the first two reviewers."""
    from statistics import mean
    import collections
    buckets = collections.defaultdict(list)
    for t in traces:
        if len(t.verdicts) >= 2:
            buckets[t.attack_family].append((t.verdicts[0], t.verdicts[1]))
    out = {}
    for fam, pairs in buckets.items():
        out[fam] = _cohens_kappa(pairs)
    return out


def _cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    # Standard two-rater kappa over nominal labels; see Cohen (1960).
    from collections import Counter
    n = len(pairs)
    if n == 0: return float("nan")
    labels = sorted({l for p in pairs for l in p})
    obs = sum(1 for a, b in pairs if a == b) / n
    pa = Counter(a for a, _ in pairs); pb = Counter(b for _, b in pairs)
    exp = sum((pa[l] / n) * (pb[l] / n) for l in labels)
    return (obs - exp) / (1 - exp) if exp < 1 else float("nan")


# ---- "chosen" response sources --------------------------------------------

def chosen_from_self_safe(trace: Trace) -> str | None:
    """Re-run the model on the same prompt with a strong safety system prompt.
    Returns a safe refusal / safe-completion. Illustrative wrapper below."""
    sys_msg = (
        "You are an AI assistant. Decline any harmful request with a short,"
        " non-moralizing refusal following the response policy in"
        " mod-106 exercise-01. If the request is benign, answer directly."
    )
    return _chat_wrapped(sys_msg, trace.dialog)


def chosen_from_cai_revision(trace: Trace) -> str | None:
    """Run exercise-02's cai_loop over the same dialog; use the final text."""
    from cai_loop import run_cai_once, _load_constitution
    constitution = _load_constitution("constitution.md")
    cai_trace = run_cai_once(trace.id, _render_prompt(trace.dialog), constitution)
    return cai_trace.final_text if cai_trace.converged else None


def chosen_from_exercise_01(prompt: str, target_harm: str) -> str | None:
    """Retrieve a nearest-neighbor refusal / safe-completion from the
    exercise-01 corpus, matched on target_harm and prompt-embedding cosine."""
    # NN lookup against exercise-01's safety_sft.jsonl; returns None if no
    # neighbor within cosine ≥ 0.6. Keeps the "chosen" distribution close to
    # a human-authored reference rather than drifting toward model outputs.
    ...


# ---- the driver -----------------------------------------------------------

def build_pair(trace: Trace, kappa: dict[str, float],
               lsh: MinHashLSH) -> Pair | None:
    verdict = majority_verdict(trace.verdicts)
    if verdict != "unsafe":
        return None                                # not a usable rejection

    rendered_prompt = _render_prompt(trace.dialog)       # multi-turn aware
    rejected = trace.dialog[-1]["content"]               # final assistant turn

    # Three chosen sources, in order of preference.
    chosen, chosen_source = None, None
    for fn, src in (
        (lambda: chosen_from_exercise_01(rendered_prompt, trace.target_harm), "exercise-01"),
        (lambda: chosen_from_cai_revision(trace),                              "cai-revised"),
        (lambda: chosen_from_self_safe(trace),                                 "self-safe"),
    ):
        candidate = fn()
        if candidate and not _is_unsafe_text(candidate, trace.target_harm):
            chosen, chosen_source = candidate, src
            break
    if not chosen:
        return None                                # could not find a safe pole

    pair = Pair(
        id=hashlib.sha1(
            (rendered_prompt + chosen + rejected).encode("utf-8")).hexdigest(),
        prompt=rendered_prompt,
        chosen=chosen, rejected=rejected,
        preference_dimension="harmless",
        attack_family=trace.attack_family,
        target_harm=trace.target_harm,
        chosen_source=chosen_source,
        rejected_source_trace=trace.id,
        agreement={"kappa_family": kappa.get(trace.attack_family, float("nan")),
                   "raters_agreeing": sum(1 for v in trace.verdicts if v == "unsafe"),
                   "n_raters": len(trace.verdicts)},
        length_tokens={"prompt":   _tok_len(rendered_prompt),
                       "chosen":   _tok_len(chosen),
                       "rejected": _tok_len(rejected)},
    )

    # Near-dup check
    m = _minhash(rendered_prompt + "\n" + rejected)
    if lsh.query(m):
        return None
    lsh.insert(pair.id, m)
    return pair


def main() -> None:
    traces = list(_load_traces())                  # 2,000 audited traces
    kappa = kappa_by_family(traces)

    # Drop whole families with kappa < 0.6 — relabel before shipping.
    allowed_families = {f for f, k in kappa.items() if k >= 0.6}
    traces = [t for t in traces if t.attack_family in allowed_families]

    lsh = MinHashLSH(threshold=0.9, num_perm=64)
    pairs: list[Pair] = []
    for t in traces:
        p = build_pair(t, kappa, lsh)
        if p is not None:
            pairs.append(p)

    # Stratified 85/10/5 split by (attack_family, target_harm).
    train, val, test = _stratified_split(pairs, (0.85, 0.10, 0.05), seed=0)

    for shard, name in [(train, "train"), (val, "val"), (test, "test")]:
        with (OUT / f"safety_prefs_{name}.jsonl").open("w") as f:
            for p in shard:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    (OUT / "kappa_by_family.json").write_text(json.dumps(kappa, indent=2))


if __name__ == "__main__":
    main()
```

### 2.3 Multi-turn trace handling

```python
def _render_prompt(dialog: list[dict]) -> str:
    """Render all turns UP TO (but not including) the final assistant turn
    using the target tokenizer's chat template. This preserves the context
    the attack built up — a jailbreak that lands in turn 4 after benign
    turns 1–3 cannot be reproduced from turn 4 alone."""
    # Keep all turns through the final USER turn; the final ASSISTANT turn
    # becomes `rejected`.
    cutoff = next(
        (i for i in reversed(range(len(dialog))) if dialog[i]["role"] == "user"),
        None)
    assert cutoff is not None
    prefix = dialog[: cutoff + 1]
    return TOK.apply_chat_template(prefix, tokenize=False,
                                   add_generation_prompt=True)
```

The pair's `prompt` is the **full dialog prefix up to and including the final user turn**; `chosen` and `rejected` are the two possible assistant responses to that prefix. Collapsing a 4-turn attack into a single-turn pair throws away the context that made the attack work and the reward model learns nothing useful.

### 2.4 `pairs_audit.md` — target shape

```markdown
# Safety Preference Pairs — Audit — v1

## Trace audit

| | Count |
|---|---:|
| Raw traces loaded | 2,134 |
| Traces with ≥ 2 reviewer labels | 2,112 |
| Traces with majority `unsafe` verdict | 1,683 |
| Traces dropped — family κ < 0.6 (A05, A09) | 211 |
| Traces usable for pair construction | 1,472 |

## Inter-annotator agreement by family

| Family | n | Cohen's κ | Action |
|---|---:|---:|---|
| A01 Direct | 342 | 0.88 | ship |
| A02 Roleplay | 198 | 0.74 | ship |
| A03 Prefix injection | 221 | 0.81 | ship |
| A04 Obfuscation | 157 | 0.69 | ship |
| A05 Language mix | 142 | 0.51 | DROP — relabel guide v2 |
| A06 Multi-turn | 303 | 0.78 | ship |
| A07 Tool wrapper | 128 | 0.72 | ship |
| A08 Hypothetical | 169 | 0.65 | ship |
| A09 System-prompt exfil | 69 | 0.42 | DROP — redefine category |

## Pair composition

| Chosen source | Pairs | Share |
|---|---:|---:|
| exercise-01 corpus (nearest-neighbor refusal) | 612 | 42% |
| cai-revised (exercise-02 loop) | 541 | 37% |
| self-safe (model + strong safety sysmsg) | 319 | 22% |
| Total | 1,472 | 100% |

(Prefer exercise-01-sourced when available: it keeps `chosen` distribution
closest to a human-authored reference rather than drifting toward
model outputs.)

## Known biases

- **Length.** `chosen` responses from exercise-01 are short refusals;
  `rejected` responses are long jailbreak completions. Length delta
  trains length-preference unless the reward model is length-controlled.
  Mitigation: pass `length_tokens` as a sample weight into exercise-02
  reward model training; or subsample long `rejected` responses.
- **Style.** `self-safe` chosen responses inherit the model's style; this
  biases the reward model to prefer that style. Mitigation: the 42%
  share from exercise-01 is a diversity floor.
- **Multi-turn under-representation.** A06 is 21% of usable traces but
  most practitioner red-team data is single-turn. Flag this gap in the
  data card and cover it with a targeted eval set (exercise-04).

## Decontamination
- Overlap against exercise-04 eval prompts (Jaccard ≥ 0.8): 0.
- Overlap against mod-102 SFT eval prompts (Jaccard ≥ 0.8): 0.

## Splits
Train / val / test = 85 / 10 / 5. Stratified by
`(attack_family, target_harm)`. Seed = 0.
```

### 2.5 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Two-rater verdicts with majority-of-2 | Single-rater red-team labels have ~30% noise on hard cases; the reward model fits the noise | Single rater | Cheaper upfront, far worse RM downstream |
| κ threshold of 0.6 per family | Below ~0.6, the "label" is barely better than chance; training on it poisons the family | No κ gate | Silent corruption of specific attack families |
| Three `chosen` sources in priority order | Each source has a different bias; mixing diversifies while keeping an exercise-01 reference anchor | Only model-generated safe responses | Chosen distribution collapses to the model's style |
| Prefer exercise-01-sourced `chosen` | Keeps the harmlessness RM anchored to human-authored refusals from a known taxonomy | Prefer CAI revisions | Model-only loop drifts toward model's own patterns |
| Preserve full multi-turn prefix in `prompt` | Multi-turn jailbreaks depend on the build-up; truncating destroys the signal | Last-turn-only | RM cannot learn to resist multi-turn attacks |
| Stratify split by `(family, target_harm)` | Prevents a single family (A06 at 21%) from dominating val or test | Random split | Family-specific over/underfitting on val |
| Drop pairs where `chosen` is itself classified unsafe | Catches the case where `self-safe` also jailbreaks on the same prompt | Keep all candidate chosens | Silently labels unsafe responses as safe |
| Length-token metadata on every row | Downstream RM MUST length-control; this ships the signal through | Omit lengths | RM learns length-is-safe correlation |
| Per-family agreement + known-bias section in audit | External reviewer can see which families to trust | Global agreement number | Global κ hides a 0.88 family averaged with a 0.42 family |

## 3. Validation steps

1. **Per-family agreement gate.** `kappa_by_family.json` reports κ ≥ 0.6 for every family in the shipped corpus. Any family below is excluded or relabeled; a global κ alone fails the gate.
2. **Two-pole safety.** For every row, re-run the pair through an independent safety classifier (e.g., Llama Guard 3 or a stronger model with the response policy). `chosen` must be labeled safe and `rejected` labeled unsafe at ≥ 95%. Failures are inspected and dropped.
3. **Length-bias readout.** Compute `mean(len(chosen)) - mean(len(rejected))` and per-family deltas. Log in `pairs_audit.md`. If the delta exceeds 1.5× in either direction, flag to the reward-model trainer for length-control.
4. **Multi-turn integrity.** For every A06 row, `prompt` must contain at least 2 `user` turns and the `rejected` must be an assistant response directly after the final user turn. Grep for malformed dialog shape.
5. **Decontam.** MinHash-LSH against exercise-04 eval prompts at Jaccard ≥ 0.8 returns 0 hits. Hard failure if not.
6. **Determinism.** `python red_team_pipeline.py --seed 0` on the same input produces byte-identical outputs twice.
7. **Position-sensitivity.** For 100 random pairs, swap `chosen ↔ rejected` and send through the independent safety classifier. The classifier must flip its verdict in ≥ 95%; stable verdicts mean the classifier is reading length or format, not content.
8. **Schema completeness.** `jq 'select(.chosen_source == null or .attack_family == null or .target_harm == null or .agreement == null)' *.jsonl` returns nothing.
9. **Stratification.** After split, each (attack_family, target_harm) bucket is represented in train/val/test proportional to its population within ±20%.
10. **Null-pair audit.** Grep for rows where `chosen == rejected` or token-overlap ≥ 0.9. Must be zero — a pair with identical poles trains nothing.

## 4. Rubric / review checklist

| # | Check | Pass criteria |
|---|---|---|
| 1 | `attack_taxonomy.md` exists, versioned, with verdict labels and agreement protocol | Yes / No |
| 2 | Every trace has ≥ 2 verdicts from distinct reviewers | ≥ 98% of usable traces |
| 3 | Cohen's κ reported per family; families with κ < 0.6 excluded | Yes / No |
| 4 | Pair schema includes `attack_family`, `target_harm`, `chosen_source`, `rejected_source_trace`, `agreement`, `length_tokens` | Yes / No |
| 5 | `chosen` is sourced from exercise-01 ≥ 30% of the time | Yes / No |
| 6 | Multi-turn pairs preserve full dialog prefix in `prompt` | 100% of A06 rows |
| 7 | Independent safety classifier agrees with the pair's polarity on ≥ 95% | Yes / No |
| 8 | MinHash-LSH decontam against exercise-04 eval returns 0 hits | Yes / No |
| 9 | Pipeline is deterministic under fixed seed | Yes / No |
| 10 | `pairs_audit.md` includes the length-bias readout and known-biases section | Yes / No |

## 5. Common mistakes

- Treating raw red-team logs as ground-truth preference pairs. The unsafe response is the `rejected`; the `chosen` has to be constructed separately and both sides must pass an independent safety check.
- Single-reviewer verdicts. Red-team labels disagree 20–30% on hard cases; training on single-rater noise gives the reward model an incoherent harmlessness objective.
- Computing a global κ. The useful breakdown is per family: a direct-attack family at κ=0.88 averaged with a language-mix family at κ=0.42 looks like a reasonable κ=0.6 but hides a family that cannot be trained on.
- Collapsing multi-turn traces to a single-turn pair. The jailbreak that worked in turn 4 was built by turns 1–3. Training on turn 4 alone teaches nothing about resisting the attack.
- One source for `chosen`. If every `chosen` is a model-generated safe response, the reward model learns to prefer the model's style, not harmlessness.
- Shipping with `chosen` distribution shorter than `rejected`. Length delta trains length-preference unless the reward model is length-controlled; log it in the audit and pass `length_tokens` through.
- Dropping the trace id. Without `rejected_source_trace`, the audit cannot walk from a shipped pair back to the red-team session that produced it — chain of custody breaks.
- Letting family coverage drift. Training mostly on direct attacks (A01) produces a model great at refusing direct attacks and no better than baseline at roleplay or multi-turn. Stratification is mandatory.
- Mixing in eval prompts. HarmBench, AdvBench, JailbreakBench, XSTest, and OR-Bench are all public; any overlap into training invalidates exercise-04's numbers.
- Treating this as a one-shot corpus. New attack families appear (code-wrapper and obfuscation attacks evolve month-to-month); version the taxonomy and the pipeline so the SFT/DPO run is pinnable to a snapshot.

## 6. References

- Ganguli et al., "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned," 2022. <https://arxiv.org/abs/2209.07858> — framing of red-team trace collection, inter-rater agreement protocols, and attack taxonomy.
- Perez et al., "Red Teaming Language Models with Language Models," 2022. <https://arxiv.org/abs/2202.03286> — automated red-team generation; source of A03/A08-style patterns and a standard protocol for scaling red-team coverage.
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback," 2022. <https://arxiv.org/abs/2212.08073> — the `cai-revised` path for `chosen` responses in this pipeline.
- Cohen, J., "A Coefficient of Agreement for Nominal Scales," 1960. <https://doi.org/10.1177/001316446002000104> — Cohen's kappa for inter-annotator agreement.
- OWASP Machine Learning Security Top 10 (ML01 Input Manipulation, ML09 Output Integrity Attack, ML10 Model Inversion). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — adversarial threat landscape for AI systems; mapping used to align A01–A09 with external tactics and techniques. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (AI RMF 1.0) — *Measure 2.6 Safe*, *Measure 2.7 Secure and Resilient*, *Manage 2.1 (document mitigations)*. The audit file and pipeline logs are the evidence artifacts for these functions. <https://www.nist.gov/itl/ai-risk-management-framework>
- `Anthropic/hh-rlhf` dataset (harmless-base split) — representative red-team trace source. <https://huggingface.co/datasets/Anthropic/hh-rlhf>
- mod-105 · exercise-01 — preference-pair schema this file follows.
- mod-106 · exercise-01 — harm taxonomy, response-mode templates, and dual-use conventions used to validate `chosen`.
- mod-106 · exercise-02 — the CAI self-critique loop used as a `chosen` source.
- mod-106 · exercise-04 — over-refusal + jailbreak eval that this training corpus must be decontaminated against.
