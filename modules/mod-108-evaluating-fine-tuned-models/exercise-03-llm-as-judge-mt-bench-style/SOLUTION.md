# mod-108 · exercise-03 — LLM-as-Judge, MT-Bench Style — Solution

## 1. Solution overview

The deliverable is a **calibrated LLM-as-judge harness** that scores model responses — either absolute (1–10 rubric) or pairwise (A vs. B vs. tie) — along with the *calibration data* that makes the scores trustworthy: inter-annotator agreement between the judge and a double-annotated human set, a position-bias test, and a self-preference test. The MT-Bench methodology (Zheng et al., 2023) is the reference template; the exercise adapts it to a task-specific eval.

Four ideas carry the exercise:

- **A judge without calibration is a random number generator.** Pairing an LLM judge with a 200–500 example double-annotated human set to compute Cohen's κ (or Krippendorff's α) is the step that converts "the judge said we won" into "the judge agrees with humans on this task, at this prompt, with this policy, κ ≈ 0.6." Without the κ, the judge output is not evidence.
- **Pairwise is more robust than absolute, but both have biases.** Pairwise asks "is A better than B?" and is less sensitive to rubric drift and anchoring. Absolute 1–10 scoring is easier to aggregate across models over time but suffers from scale compression (judges rarely use 1–3 or 9–10) and prompt-dependent ceilings. The harness supports both and documents the tradeoff.
- **Judges have positional, verbosity, and self-preference biases that must be measured.** Randomize A/B order per pair; test by running the same pair under both orderings and asking whether the verdict is consistent. Verbosity bias is checked by sampling matched-rubric-score pairs where one is 1.5× longer. Self-preference is checked by including the judge model's own outputs in the pool.
- **The judge prompt is a product, not an afterthought.** Rubric wording, role framing, chain-of-thought placement, and the output schema (JSON vs. free text) all move numbers by multiple points. The prompt is versioned and the version is attached to every judgment.

The worked example scores customer-support summaries (the fine-tune target from exercise 02) using GPT-4-class and Claude-class judges, with calibration against a 400-row human-annotated set.

## 2. Worked implementation

### 2.1 Judge modes

Two modes supported, same harness:

- **Pairwise (preferred for comparisons).** For every eval prompt, the judge sees responses from two models (A, B) and returns `winner ∈ {A, B, tie}` with a brief rationale. The harness runs each pair twice — once as (A,B) and once as (B,A) — and reports the win-rate with a *consistency* column that flags pairs whose verdict flipped under position swap.
- **Absolute (preferred for tracking over time).** The judge sees one response plus the task spec and rubric and returns a per-dimension 1–10 score with rationale. Rubric dimensions match the held-out spec from exercise 02.

### 2.2 The pairwise judge prompt (versioned)

```python
PAIRWISE_PROMPT_V1_0 = """You are evaluating two AI-written customer-support summaries, A and B, against the same source transcript. Decide which is better, or declare a tie.

Judge on these dimensions, in this order:
1. Accuracy: does the summary reflect what actually happened in the transcript? No invented facts, no misattributed statements.
2. Completeness: does it cover the customer's issue, the agent's resolution, and any open follow-ups?
3. Hallucination: does it contain any named entity, number, or detail not present in the source?
4. Length discipline: is it 60–120 words?
5. PII suppression: does it omit email addresses, phone numbers, SSNs, credit-card digits?

Evaluate silently, then output ONLY this JSON (no prose outside):

{{
  "accuracy_winner": "A|B|tie",
  "completeness_winner": "A|B|tie",
  "hallucination_winner": "A|B|tie",
  "length_winner": "A|B|tie",
  "pii_winner": "A|B|tie",
  "overall_winner": "A|B|tie",
  "rationale": "one or two sentences, 30 words max",
  "confidence": "low|medium|high"
}}

A tie is appropriate only when both summaries are nearly interchangeable on that dimension. Do not use tie to avoid making a decision.

Source transcript:
<<<
{source}
>>>

Summary A:
<<<
{response_a}
>>>

Summary B:
<<<
{response_b}
>>>
"""
```

The versioning convention: `_V<MAJOR>_<MINOR>`. A minor bump changes wording only (clarification, typo). A major bump changes the rubric, the dimensions, or the output schema; prior numbers do not carry forward to the new version.

### 2.3 The absolute judge prompt

```python
ABSOLUTE_PROMPT_V1_0 = """Score the following customer-support summary on five dimensions. Each dimension is scored 1–5 where:
  5 = flawless on this dimension
  4 = minor issue, still acceptable
  3 = a reviewer would ask for revision
  2 = unacceptable
  1 = severe (e.g. an invented fact or a leaked phone number)

Dimensions:
  - accuracy: no misattributed statements, no invented facts
  - completeness: issue, resolution, follow-ups all covered
  - hallucination: no entities/numbers/details absent from source
  - length: 60–120 words
  - pii: no emails, phone numbers, SSNs, credit-card digits

Output ONLY this JSON:
{{
  "accuracy": 1-5,
  "completeness": 1-5,
  "hallucination": 1-5,
  "length": 1-5,
  "pii": 1-5,
  "rationale": "one or two sentences",
  "confidence": "low|medium|high"
}}

Source transcript:
<<<
{source}
>>>

Summary:
<<<
{response}
>>>
"""
```

The 1–5 Likert is preferred over 1–10 for this task. MT-Bench uses 1–10, but on a narrow task (summarization of a redacted transcript) the 5-point scale produces calibrated distributions; 1–10 tends to cluster at 7–8 with little information per point.

### 2.4 The judge harness

```python
# judge.py — pairwise or absolute scoring, with position randomization.
from __future__ import annotations
import json, hashlib, pathlib, random, time
from dataclasses import dataclass, asdict
from typing import Literal

@dataclass
class PairJudgment:
    pair_id: str                       # deterministic from (doc_id, model_a, model_b)
    doc_id: str
    model_a: str                       # model whose output is in position A for this row
    model_b: str
    forward_winner: str                # "A" | "B" | "tie"
    reverse_winner: str                # same judgment with A/B swapped
    consistent: bool                   # forward_winner == swapped(reverse_winner)
    prompt_version: str
    judge_model: str
    rationale_forward: str
    rationale_reverse: str
    confidence_forward: str
    confidence_reverse: str

def _pair_id(doc_id: str, model_a: str, model_b: str) -> str:
    a, b = sorted([model_a, model_b])
    return hashlib.sha256(f"{doc_id}:{a}:{b}".encode()).hexdigest()[:16]

def _swap(winner: str) -> str:
    return {"A": "B", "B": "A", "tie": "tie"}[winner]

def pairwise(doc: dict, resp_a: str, resp_b: str, model_a: str, model_b: str,
             judge_call, prompt_tmpl: str, prompt_version: str,
             judge_model: str) -> PairJudgment:
    # Forward: (A, B)
    fwd = judge_call(prompt_tmpl.format(source=doc["source"], response_a=resp_a, response_b=resp_b))
    # Reverse: swap positions, do NOT just swap the output; re-run to catch position bias
    rev = judge_call(prompt_tmpl.format(source=doc["source"], response_a=resp_b, response_b=resp_a))
    return PairJudgment(
        pair_id=_pair_id(doc["doc_id"], model_a, model_b),
        doc_id=doc["doc_id"],
        model_a=model_a, model_b=model_b,
        forward_winner=fwd["overall_winner"],
        reverse_winner=_swap(rev["overall_winner"]),       # re-map to A's identity
        consistent=(fwd["overall_winner"] == _swap(rev["overall_winner"])),
        prompt_version=prompt_version,
        judge_model=judge_model,
        rationale_forward=fwd["rationale"],
        rationale_reverse=rev["rationale"],
        confidence_forward=fwd["confidence"],
        confidence_reverse=rev["confidence"],
    )

def pairwise_verdict(j: PairJudgment) -> str:
    """Reduce two judgments to one with position-bias handling."""
    f, r = j.forward_winner, j.reverse_winner
    if f == r:                                             # both runs agree
        return f
    # Disagreement → tie, but flag it
    return "tie_disputed"

def aggregate(judgments: list[PairJudgment]) -> dict:
    n = len(judgments)
    wins_a = sum(1 for j in judgments if pairwise_verdict(j) == "A")
    wins_b = sum(1 for j in judgments if pairwise_verdict(j) == "B")
    ties   = sum(1 for j in judgments if pairwise_verdict(j) in ("tie", "tie_disputed"))
    disputed = sum(1 for j in judgments if pairwise_verdict(j) == "tie_disputed")
    from math import sqrt
    # Two-sided Wilson CI on win rate (count ties as 0.5 for the headline).
    k = wins_a + 0.5 * ties
    p_hat = k / max(1, n)
    se = sqrt(p_hat * (1 - p_hat) / max(1, n))
    return {
        "n_pairs": n,
        "A_wins": wins_a, "B_wins": wins_b,
        "ties": ties, "disputed_ties": disputed,
        "A_win_rate_ties_as_half": p_hat,
        "A_win_rate_ci95": [max(0, p_hat - 1.96*se), min(1, p_hat + 1.96*se)],
        "position_consistency": sum(1 for j in judgments if j.consistent) / max(1, n),
    }
```

### 2.5 Calibration against humans

```python
# calibrate.py — compute Cohen's κ between the judge and human annotators.
from __future__ import annotations
import json, pathlib
from collections import Counter

LABELS = ("A", "B", "tie")

def cohen_kappa(h_labels: list[str], j_labels: list[str]) -> float:
    assert len(h_labels) == len(j_labels)
    n = len(h_labels)
    po = sum(1 for h, j in zip(h_labels, j_labels) if h == j) / n
    h_counts = Counter(h_labels)
    j_counts = Counter(j_labels)
    pe = sum((h_counts[k] / n) * (j_counts[k] / n) for k in LABELS)
    return (po - pe) / max(1e-12, 1 - pe)

def confusion_matrix(h: list[str], j: list[str]) -> dict:
    m = {a: {b: 0 for b in LABELS} for a in LABELS}
    for hi, ji in zip(h, j):
        m[hi][ji] += 1
    return m

def run(judge_file: pathlib.Path, human_file: pathlib.Path) -> None:
    """Both files are jsonl of pairwise judgments; keyed on pair_id."""
    judge  = {d["pair_id"]: d for d in _read_jsonl(judge_file)}
    human  = {d["pair_id"]: d for d in _read_jsonl(human_file)}
    common = sorted(set(judge) & set(human))
    h_labels = [human[p]["overall_winner"] for p in common]
    j_labels = [judge[p]["forward_winner"] for p in common]  # or pairwise_verdict(...)
    kappa = cohen_kappa(h_labels, j_labels)
    agreement = sum(1 for h, j in zip(h_labels, j_labels) if h == j) / len(common)
    cm = confusion_matrix(h_labels, j_labels)
    report = {
        "n": len(common),
        "cohen_kappa": round(kappa, 3),
        "raw_agreement": round(agreement, 3),
        "confusion_matrix": cm,
        "interpretation": _kappa_interp(kappa),
    }
    pathlib.Path("results/calibration.json").write_text(json.dumps(report, indent=2))

def _kappa_interp(k: float) -> str:
    # Landis & Koch 1977 thresholds — widely used though not universally endorsed
    if k < 0.0:  return "worse than chance"
    if k < 0.2:  return "slight"
    if k < 0.4:  return "fair"
    if k < 0.6:  return "moderate"
    if k < 0.8:  return "substantial"
    return "almost perfect"
```

The target κ for a judge that can replace most human grading on this task is **≥ 0.6 (substantial agreement)** per Landis & Koch 1977. A κ below 0.4 ("fair") means the judge is useful as a prefilter but every judgment must still go to a human for the release decision; the harness must gate the use-case on the measured κ.

### 2.6 Bias tests

```python
# bias_tests.py — position, verbosity, and self-preference checks.
from __future__ import annotations
import json, pathlib, random
from statistics import mean

def position_bias(judgments: list[dict]) -> dict:
    """Does the judge favour position A regardless of content?
    Expectation under no bias: among single-run forward judgments, A wins == B wins.
    We use the full dataset's forward verdicts only."""
    n = len(judgments)
    a = sum(1 for j in judgments if j["forward_winner"] == "A")
    b = sum(1 for j in judgments if j["forward_winner"] == "B")
    ties = n - a - b
    # Binomial two-sided p-value on {A,B} only (ties dropped)
    from math import comb
    m = a + b
    k = min(a, b)
    p = 2 * sum(comb(m, i) * 0.5**m for i in range(k + 1))
    return {"A_wins": a, "B_wins": b, "ties": ties, "p_value_two_sided": round(p, 4)}

def verbosity_bias(judgments: list[dict], outputs: dict[str, str]) -> dict:
    """On pairs where human-labeled quality is a tie, does the judge prefer the longer one?"""
    tie_pairs = [j for j in judgments if j.get("human_label") == "tie"]
    longer_wins = 0
    for j in tie_pairs:
        wc_a = len(outputs[j["pair_id"] + ":A"].split())
        wc_b = len(outputs[j["pair_id"] + ":B"].split())
        winner = j["forward_winner"]
        if winner == "A" and wc_a > wc_b: longer_wins += 1
        elif winner == "B" and wc_b > wc_a: longer_wins += 1
        elif winner == "tie": pass
    return {
        "n_human_tie_pairs": len(tie_pairs),
        "longer_wins_of_decided": longer_wins,
    }

def self_preference(judgments: list[dict], judge_model: str) -> dict:
    """In pairs where one side was generated by the judge model itself,
    does the judge disproportionately pick its own output?"""
    self_in_a = [j for j in judgments if j["model_a"] == judge_model]
    self_in_b = [j for j in judgments if j["model_b"] == judge_model]
    sw_a = sum(1 for j in self_in_a if j["forward_winner"] == "A")
    sw_b = sum(1 for j in self_in_b if j["forward_winner"] == "B")
    n_self = len(self_in_a) + len(self_in_b)
    wins_for_self = sw_a + sw_b
    return {
        "n_self_pairs": n_self,
        "wins_for_self": wins_for_self,
        "self_win_rate": wins_for_self / max(1, n_self),
    }
```

### 2.7 Worked results (illustrative)

Reference numbers from running the harness with Claude 3.5 Sonnet-class judge on 1,200 pairs from exercise 02, with 400 double-annotated by humans. Treat the digits as illustrative; the claim is the *shape* of the output.

```json
{
  "pairwise_aggregate": {
    "n_pairs": 1200,
    "A_wins": 684, "B_wins": 416,
    "ties": 100, "disputed_ties": 44,
    "A_win_rate_ties_as_half": 0.611,
    "A_win_rate_ci95": [0.584, 0.638],
    "position_consistency": 0.906
  },
  "calibration": {
    "n": 400,
    "cohen_kappa": 0.63,
    "raw_agreement": 0.78,
    "interpretation": "substantial"
  },
  "position_bias": {
    "A_wins": 612, "B_wins": 509, "ties": 79,
    "p_value_two_sided": 0.0021
  },
  "verbosity_bias": {
    "n_human_tie_pairs": 74,
    "longer_wins_of_decided": 48
  },
  "self_preference": {
    "n_self_pairs": 220,
    "wins_for_self": 128,
    "self_win_rate": 0.582
  }
}
```

**Reading the output**:

- Headline: tuned (A) beats base (B) 61.1% of the time, CI95 [58.4%, 63.8%]; the lower bound excludes 50%, so the win is real.
- Position consistency is 90.6% — the judge agrees with itself under A/B swap on 9 of 10 pairs. 10% of pairs show position-sensitive judgments; these are reported as `tie_disputed` rather than counted as wins.
- κ = 0.63 ("substantial") — the judge can be used as a primary grader for most of this task's eval, with humans spot-checking ambiguous cases.
- Position bias: p = 0.002 — there is a statistically significant preference for position A in the raw forward judgments (612 A vs. 509 B, with 79 ties). This is why the harness runs every pair in both orderings.
- Verbosity bias: on 74 human-tied pairs, the judge picked the longer answer 48/74 = 65% of the time. Noticeable; disclosed in the report.
- Self-preference: 58.2% self-win-rate against other models in the pool; expected to be > 50% because the judge was independently evaluated as above-average on the task, but worth isolating — the release decision uses a non-judge model's outputs or a different judge for the self-pairs.

### 2.8 `report.md` skeleton

```markdown
# LLM-as-judge report — tuned vs. base (1,200-pair run)

## Setup
- Judge: anthropic/claude-3-5-sonnet-20241022 (temperature 0, top_p 1)
- Prompt: PAIRWISE_PROMPT_V1_0 (SHA 1a2b3c…)
- Position-randomized: yes, every pair run as (A,B) and (B,A)
- Calibration set: 400 double-annotated pairs; κ = 0.63 (substantial)

## Headline
Tuned model wins 61.1% of the 1,200 pairs; CI95 [58.4%, 63.8%]. Lower bound
excludes 50% → this is a genuine win at α=0.05.

## Position / verbosity / self-preference
- Raw forward judgments show a 54.5% preference for position A (p=0.002).
  Mitigated by the two-run-with-swap pipeline; headline uses the reconciled
  verdict, not the forward run.
- Judge picks the longer answer on 48 of 74 human-tied pairs — a
  non-trivial verbosity bias. Reported, not corrected; downstream decisions
  that care about terseness need a different judge (or add length to the
  prompt as a *penalty* dimension).
- Self-preference against this judge's own model is 58.2%. The self-pairs
  are excluded from the headline tuned-vs-base comparison (neither is the
  judge's own model); flagged for future care.

## Qualitative patterns (from rationales)
- Judge consistently penalizes the base model for omitting the "open follow-ups" bullet.
- Both models occasionally hallucinate dates; the judge catches these at
  κ=0.71 on the hallucination sub-dimension vs. κ=0.63 overall.

## Limitations
- Calibration set is 400 pairs from the same distribution as the full eval;
  OOD performance (e.g. different language, different issue type) is not
  separately calibrated.
- Verbosity bias: for release gating on terseness-critical tasks, a separate
  deterministic length check is required.
```

### 2.9 Directory layout

```
outputs/exercise-03/
├── prompts/
│   ├── pairwise_v1_0.py
│   └── absolute_v1_0.py
├── judge.py
├── calibrate.py
├── bias_tests.py
├── data/
│   ├── pairs.jsonl                     # 1,200 eval-set pairs
│   └── human_labels.jsonl              # 400 double-annotated
├── results/
│   ├── judgments.jsonl                 # one row per pair, both directions
│   ├── aggregate.json
│   ├── calibration.json
│   ├── bias.json
│   └── report.md
└── README.md
```

## 3. Validation steps

1. **Every judgment is run in both orderings.** Grep the `judgments.jsonl` for any pair with only one row; a single-direction judgment is unusable for the reconciled verdict and is a hard fail.
2. **Prompt version is attached to every row.** The `prompt_version` and `judge_model` fields appear on every judgment. A missing version means the run cannot be reproduced after the prompt is edited.
3. **Position consistency ≥ 85%.** Below 85% the judge is being driven by position, not content, on a significant fraction of pairs; the harness is not useful as a comparator until the prompt is revised to reduce position sensitivity.
4. **Calibration κ reported and above 0.4.** κ < 0.4 means the judge disagrees with humans more than it agrees above chance; use it only as a prefilter, do not report win rates as if they were ground truth.
5. **Calibration set is double-annotated.** The human labels include two independent annotators per pair; inter-human κ is reported in `calibration.json`. Single-annotator "ground truth" is an oxymoron.
6. **Bias tests are run and reported.** `bias.json` contains all three (position, verbosity, self-preference). A judge that passes calibration but shows severe position bias has produced win rates that are not reproducible; the report must disclose.
7. **Temperature is 0 for judgments.** Any non-zero temperature introduces noise; this is a comparator, not a generator. Grep the API-call code for the temperature parameter.
8. **Self-pairs are excluded from the headline.** If the judge model generates one of the sides in a comparison, that pair is removed from the aggregate and reported in a separate line.
9. **Rationales are stored.** Every judgment has a rationale string; downstream triage reads rationales to find patterns (e.g. "judge always penalizes the base for missing the follow-ups bullet") that are what the human reviewer acts on.
10. **CI on the headline win rate excludes 50%** before any "win" is claimed. The harness emits the CI; the reviewer reads it.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Prompt and schema quality** | Free-text output; no schema; no versioning | JSON schema but unversioned; rubric not decomposed | Versioned prompt; JSON schema with per-dimension winners; decomposed rubric; `confidence` field present |
| **Position-bias handling** | Single-direction judgments only | Two-direction but no consistency flag | Every pair run in both orderings; `consistent` flag per pair; `tie_disputed` category in aggregation |
| **Calibration against humans** | Not attempted | κ computed but calibration set < 100 or single-annotated | ≥ 200 double-annotated calibration pairs, κ reported with Landis-Koch interpretation, confusion matrix emitted |
| **Bias testing** | None | Position bias only | Position + verbosity + self-preference, each with numeric result and disclosure in the report |
| **Report defensibility** | A single win-rate number | Win-rate + CI | Win-rate + CI, calibration κ, all three bias results, qualitative patterns from rationales, named limitations |

**Pass threshold:** 7/10, with no single dimension at 0 and calibration at ≥ 1.

## 5. Common mistakes

- **Running every pair once and trusting the forward direction.** MT-Bench itself documents the position bias; single-direction judgments are not a comparator, they are a mood ring.
- **Reporting a win-rate without a CI.** +5% on 100 pairs is inside the CI; +5% on 10,000 pairs is massive. The width of the CI is the whole point.
- **Judging at temperature > 0.** The judge is not a generator; its output should be deterministic given the prompt. Any non-zero temperature means a rerun produces different numbers, and the κ is a lower bound on true agreement.
- **Using the same model as both judge and generator and then claiming a win.** The self-preference test catches this; the harness must either pick a different judge family (Claude judging GPT outputs, or vice versa) or exclude self-pairs.
- **No calibration.** The MT-Bench paper itself reports ~80% agreement between GPT-4 and humans on their task; your task will be different. Build the calibration set *first*, measure κ, then decide whether the judge is trusted.
- **"Overall score" as a single field.** The judge is much more useful when it decomposes: asking for per-dimension winners surfaces disagreement patterns (judge might agree on accuracy at κ=0.75 and on length at κ=0.45) that drive prompt revisions.
- **Rubric wording that leaks the expected answer.** "Which summary better reflects the agent's professional handling of the conversation?" primes the judge toward one answer style. The rubric is operational, not aspirational.
- **Ignoring the verbosity bias.** Every public LLM-judge study has documented it; a terse-correct answer can lose to a verbose-slightly-wrong answer. For terseness-critical tasks, pair the judge with a deterministic length check and report both.
- **Using a judge trained on a different format than your eval.** If the judge was tuned to grade MT-Bench-style chat turns and you give it a 20-turn customer-support transcript to grade, you will get calibration numbers that look reasonable on the surface (κ ≈ 0.4) but hide specific-dimension failures. The calibration set must come from the actual eval distribution.
- **Not storing rationales.** The numbers in `aggregate.json` tell you *that* a model won; the rationales in `judgments.jsonl` tell you *why*. Rationales are the raw material for prompt revisions on the generator and the rubric.

## 6. References

- Zheng et al. — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. The original MT-Bench paper; introduces pairwise, absolute, position/verbosity/self-preference biases, and reports GPT-4 vs. human agreement on their task. <https://arxiv.org/abs/2306.05685>
- Chatbot Arena / LMSYS — crowd-sourced pairwise human comparisons; the Elo/BT-model framework. <https://lmarena.ai/>
- FastChat — reference MT-Bench implementation (judge prompts, aggregation code). <https://github.com/lm-sys/FastChat>
- Cohen (1960) and Landis & Koch (1977) — Cohen's κ and the Landis-Koch interpretation table used in `_kappa_interp`. <https://doi.org/10.1177/001316446002000104>, <https://www.jstor.org/stable/2529310>
- Krippendorff (2004) — Krippendorff's α, the alternative agreement statistic when more than two annotators or non-nominal labels are involved. <https://repository.upenn.edu/asc_papers/43/>
- NIST AI Risk Management Framework — *Measure 2.3* and *Measure 3.2* cover trusted-evaluation practices; LLM-as-judge is a measurement instrument and must be calibrated under *Measure 2.7*. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — *ML06 AI Supply Chain Vulnerabilities* applies to pinning the judge model and prompt version in the release record; an unversioned judge is an unreproducible gate. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — *Verify Attack* and *Discover ML Model Family* categories inform the self-preference check; a judge preferring its own family's outputs is a known failure mode. <https://atlas.mitre.org/>
