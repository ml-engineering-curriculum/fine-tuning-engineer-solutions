# mod-108 · exercise-02 — Task-Specific Held-Out Eval Design — Solution

## 1. Solution overview

The deliverable is a **held-out evaluation set spec, dataset, and scoring harness** for the fine-tune's intended task — the thing the lm-eval public benchmarks (exercise 01) cannot measure because they were not designed for it. Four artifacts come out of this exercise:

- `EVAL_SPEC.md` — the written specification: what the model is being evaluated on, by which metric, against which threshold, with what statistical power.
- `data/eval_v1.jsonl` — the frozen held-out examples with stratification labels.
- `score.py` — a scoring harness that turns a model's outputs into the metrics in the spec.
- `power_analysis.md` — the sample-size calculation and documented limitations.

Four ideas carry the exercise:

- **A held-out eval is a *written contract*, not a scraped sample.** The spec names the task, the population, the metric, the pass threshold, and the statistical confidence required. If any of those is missing, the eval cannot gate a release — only produce a number that reads however the reviewer wants.
- **Stratify the eval set along the dimensions that matter for the product.** A 1000-example eval that is 95% English short prompts cannot tell you anything about the 5% long multi-turn Spanish prompts that are 40% of live traffic. Stratification is the difference between a representative eval and a convenient one.
- **Decontaminate against training data *by construction*, not after the fact.** Hash-based decontamination (exercise 04) is a backstop. The primary defense is a dataset-build process that pulls eval examples from a time-partitioned or source-partitioned slice that the training pipeline is forbidden to see.
- **Power-analyze before collecting.** An eval set sized without a power calculation either wastes labeling budget (too large) or fails to detect the release gate (too small). The exercise asks for a specific number of examples per stratum to detect a target effect size with 80% power.

The worked example designs a held-out eval for a customer-support summarization fine-tune. The pattern generalizes to classification, generation, structured-extraction, and tool-use tasks; a §2.7 adaptation table maps the template to each.

## 2. Worked implementation

### 2.1 `EVAL_SPEC.md` — the written contract

```markdown
# Held-out evaluation spec — Customer-support summarization v1

## 1. Task
Given a transcript of a customer-support conversation (1–20 turns), produce a
summary that captures the customer's issue, the agent's resolution, and any
open follow-ups. Target length 60–120 words.

## 2. Population
Production customer-support transcripts from 2026-10-01 through 2026-10-31,
redacted per the production PII policy. English and Spanish, ratio 70/30 as
measured in live traffic for the same period.

## 3. Train/eval partition
Training data is drawn from 2026-01-01 through 2026-09-30 only. The eval slice
is the final month of the quarter, held physically separate in a different
bucket with ACLs that forbid the training pipeline's service account. This is
a *time-based* partition; see exercise 04 for the decontamination check.

## 4. Primary metric
Rubric score (0–5) from a 5-dimension rubric (accuracy, completeness,
hallucination, length discipline, PII suppression), graded by human
annotators on a double-annotated subset (κ reported) and by an LLM judge
(calibrated against the human labels; see exercise 03) on the rest.
Primary release-gate metric: mean rubric score, 95% CI, and the fraction of
responses scoring ≥ 4 on every dimension ("acceptable rate").

## 5. Secondary metrics
- ROUGE-L and BERTScore against a human reference summary (reference quality
  caveats documented).
- Hallucination rate: fraction of responses where any named entity in the
  summary does not appear in the source transcript (deterministic check).
- PII leakage rate: regex + NER sweep for any PII type the redaction layer
  should have removed.
- Length-discipline rate: fraction of responses in the [60, 120] word window.

## 6. Pass threshold (release gate)
- Mean rubric ≥ 3.8 (out of 5), 95% CI lower bound ≥ 3.6.
- Acceptable rate ≥ 70%.
- Hallucination rate ≤ 2%.
- PII leakage rate ≤ 0.1% (a hard gate; a single confirmed leak fails release).
- Length-discipline rate ≥ 90%.
- No per-stratum metric regresses more than 5% absolute vs. the current
  production model's last-run numbers.

## 7. Strata (populations we need the metric on, independently)
- Language: en, es (70/30 by prior traffic).
- Transcript length: short (1–3 turns), medium (4–10), long (11–20).
- Issue type: billing, technical, account, refunds, abuse.
- Resolution outcome: resolved-first-contact, escalated, unresolved.

## 8. Sample size
See §2.3 power analysis. Target 1,200 examples total; 400 are
double-annotated for inter-annotator agreement (κ) and judge calibration.

## 9. Freeze
Dataset version v1.0 is frozen on 2026-11-20. Any change to prompts, filters,
or rubric bumps the version; prior results do not carry forward.
```

### 2.2 Stratification and sampling

```python
# build_eval.py — sample a stratified held-out set and freeze it.
from __future__ import annotations
import hashlib, json, random, pathlib
from collections import Counter

RNG = random.Random(20261120)                   # seed = freeze date
LANGS = ["en", "es"]
LANG_WEIGHTS = {"en": 0.70, "es": 0.30}
LENGTH_BINS = {"short": (1, 3), "medium": (4, 10), "long": (11, 20)}
ISSUE_TYPES = ["billing", "technical", "account", "refunds", "abuse"]
RESOLUTION = ["resolved", "escalated", "unresolved"]

TARGET_N = 1200

def _stratum_key(doc: dict) -> tuple[str, str, str, str]:
    turns = len(doc["turns"])
    bin_ = next(name for name, (lo, hi) in LENGTH_BINS.items() if lo <= turns <= hi)
    return (doc["lang"], bin_, doc["issue_type"], doc["resolution"])

def _deterministic_sort_key(doc: dict) -> str:
    # Hash the raw transcript, not a redacted or truncated version.
    return hashlib.sha256(doc["raw_transcript"].encode("utf-8")).hexdigest()

def _train_eval_split_by_time(all_docs: list[dict]) -> tuple[list[dict], list[dict]]:
    # Eval is any document from October; training is only prior months.
    train, eval_ = [], []
    for d in all_docs:
        bucket = train if d["created_at"] < "2026-10-01" else eval_
        bucket.append(d)
    return train, eval_

def build() -> None:
    all_docs = _load_production_transcripts()     # from the redacted eval bucket
    _, eval_pool = _train_eval_split_by_time(all_docs)

    # Target cell sizes: weighted by language, uniform across the other three.
    cells: dict[tuple, int] = {}
    per_language = {lang: int(round(TARGET_N * LANG_WEIGHTS[lang])) for lang in LANGS}
    for lang, n_lang in per_language.items():
        per_cell = max(1, n_lang // (len(LENGTH_BINS) * len(ISSUE_TYPES) * len(RESOLUTION)))
        for bin_ in LENGTH_BINS:
            for issue in ISSUE_TYPES:
                for res in RESOLUTION:
                    cells[(lang, bin_, issue, res)] = per_cell

    # Populate each cell deterministically.
    sampled: list[dict] = []
    shortages: list[tuple] = []
    by_cell = _group(eval_pool, _stratum_key)
    for cell, want in cells.items():
        have = sorted(by_cell.get(cell, []), key=_deterministic_sort_key)
        if len(have) < want:
            shortages.append((cell, len(have), want))
            sampled.extend(have)
        else:
            sampled.extend(have[:want])

    # Emit with a stable doc_id derived from the content hash so re-runs
    # pick the same documents even if the raw ordering drifts.
    with pathlib.Path("data/eval_v1.jsonl").open("w") as f:
        for d in sampled:
            d["doc_id"] = _deterministic_sort_key(d)[:16]
            d["stratum"] = dict(zip(("lang", "length_bin", "issue_type", "resolution"),
                                    _stratum_key(d)))
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    _write_manifest(sampled, shortages)

def _group(docs, keyfn):
    out: dict = {}
    for d in docs:
        out.setdefault(keyfn(d), []).append(d)
    return out

def _write_manifest(sampled, shortages):
    counts = Counter(tuple(d["stratum"].values()) for d in sampled)
    manifest = {
        "version": "1.0",
        "frozen_on": "2026-11-20",
        "seed": 20261120,
        "n": len(sampled),
        "by_stratum": {str(k): v for k, v in counts.most_common()},
        "shortages": [{"cell": list(c), "have": h, "want": w} for c, h, w in shortages],
    }
    pathlib.Path("data/manifest.json").write_text(json.dumps(manifest, indent=2))
```

Key design choices:

- **Deterministic sort by content hash.** Replaces "random sampling with seed" because the eval pool itself can grow between runs (new documents arrive). Content-hash sort means the same documents are picked for the same cell as long as they are present, independent of pool order or growth.
- **Separate cell quotas, not global quota.** A naive weighted-random draw from the pool would produce an eval set whose empirical distribution matches the pool, not the target traffic distribution. Stratified allocation gives every cell at least the minimum needed for a per-cell metric (see §2.3).
- **Record shortages explicitly.** If `abuse-long-unresolved-es` has only 8 documents but the quota is 25, the manifest records it. A later analysis cannot claim statistical coverage on that cell.

### 2.3 Power analysis — `power_analysis.md`

```markdown
# Sample-size calculation

## Goal
Detect a ≥ 3% absolute improvement in acceptable-rate (currently ~70%) with
80% power at α=0.05 (two-sided).

## Model
A one-sample proportion test against a baseline of 0.70; the release-gate
check is actually two-population (baseline vs. candidate) so we design the
sample size for the harder two-sample case.

n_per_group = ((z_{α/2}·√(2·p̄(1-p̄)) + z_β·√(p1(1-p1) + p2(1-p2))) / δ)²

With p1 = 0.70, p2 = 0.73, p̄ = 0.715, z_{α/2}=1.96, z_β=0.84, δ=0.03:

n_per_group ≈ (1.96·0.639 + 0.84·0.638)² / 0.03²
           ≈ (1.252 + 0.536)² / 0.0009
           ≈ 3.196 / 0.0009
           ≈ 3551

That is the per-side sample size to detect a 3-pt absolute lift on the
*overall* acceptable-rate metric. Our eval budget is one side (the candidate)
compared against a *fixed* reference (the previous production model's
sealed-dataset numbers), so we need n ≈ 3551 single-side.

## We're going to bias n down and widen δ
Labeling budget is 1200. With n=1200 and p=0.70, the SE is √(0.7·0.3/1200) ≈
0.013; the minimum detectable effect at 80% power is δ ≈ 2.8·SE ≈ 3.7%
absolute. The release gate (§2.1.6) is therefore worded as "≥ 70% and no
regression > 5%", not "detect any 1% movement."

## Per-stratum power
With 30 cells (2 lang × 3 length × 5 issue × 3 resolution) and 1200 examples,
mean cell size = 40. Per-cell SE at p=0.70 is √(0.7·0.3/40) ≈ 0.072 → cell
MDE ≈ 20 pts absolute. Per-cell metrics are *descriptive* (direction and
magnitude), not gated. Only the aggregate metric gates release.

## What we would need to gate per-stratum
To detect a 5% cell-level regression at 80% power: n_per_cell ≈ 500.
That is 15,000 examples total — 12× the current budget. Documented as a
future expansion if cell-level gates become necessary.
```

### 2.4 `score.py` — the scoring harness

```python
# score.py — turns model outputs into the metrics named in the spec.
from __future__ import annotations
import json, math, pathlib, re
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class ScoredRow:
    doc_id: str
    stratum: dict
    output: str
    rubric_mean: float            # 0..5
    rubric_breakdown: dict        # {dimension: score}
    rubric_grader: str            # "human" | "judge"
    rouge_l: float
    bertscore_f1: float
    hallucinated_entities: list[str]
    pii_matches: list[dict]
    word_count: int
    within_length_band: bool
    acceptable: bool              # every dimension >= 4

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def _hallucinated_entities(source: str, output: str) -> list[str]:
    """Any proper-noun-like span in `output` not present in `source`."""
    from_spacy = _extract_named_entities(output)
    return [e for e in from_spacy if e.lower() not in source.lower()]

PII_PATTERNS = {
    "email":    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone_us": re.compile(r"\b(?:\+?1[-\s.]?)?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}\b"),
    "ssn":      re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}

def _pii_matches(output: str) -> list[dict]:
    hits = []
    for name, rx in PII_PATTERNS.items():
        for m in rx.finditer(output):
            hits.append({"type": name, "span": m.span()})
    return hits

def _within_length_band(output: str, lo: int = 60, hi: int = 120) -> tuple[bool, int]:
    wc = len(output.split())
    return (lo <= wc <= hi, wc)

def score_file(outputs_jsonl: pathlib.Path, reference_jsonl: pathlib.Path,
               judge_scores: pathlib.Path, human_scores: pathlib.Path) -> None:
    refs = {d["doc_id"]: d for d in _read_jsonl(reference_jsonl)}
    humans = {d["doc_id"]: d for d in _read_jsonl(human_scores)}
    judges = {d["doc_id"]: d for d in _read_jsonl(judge_scores)}

    scored: list[ScoredRow] = []
    for d in _read_jsonl(outputs_jsonl):
        doc_id = d["doc_id"]
        ref = refs[doc_id]
        rubric = humans.get(doc_id) or judges.get(doc_id)
        grader = "human" if doc_id in humans else "judge"
        halluc = _hallucinated_entities(ref["source"], d["output"])
        pii = _pii_matches(d["output"])
        within, wc = _within_length_band(d["output"])
        acceptable = all(v >= 4 for v in rubric["breakdown"].values()) \
                     and len(halluc) == 0 and len(pii) == 0 and within
        scored.append(ScoredRow(
            doc_id=doc_id, stratum=ref["stratum"],
            output=d["output"],
            rubric_mean=rubric["mean"],
            rubric_breakdown=rubric["breakdown"],
            rubric_grader=grader,
            rouge_l=_rouge_l(d["output"], ref["summary"]),
            bertscore_f1=_bertscore_f1(d["output"], ref["summary"]),
            hallucinated_entities=halluc,
            pii_matches=pii,
            word_count=wc,
            within_length_band=within,
            acceptable=acceptable,
        ))
    _emit_reports(scored)

def _emit_reports(rows: list[ScoredRow]) -> None:
    n = len(rows)
    overall = {
        "n": n,
        "mean_rubric": sum(r.rubric_mean for r in rows) / n,
        "acceptable_rate": _rate(rows, lambda r: r.acceptable),
        "hallucination_rate": _rate(rows, lambda r: len(r.hallucinated_entities) > 0),
        "pii_leakage_rate": _rate(rows, lambda r: len(r.pii_matches) > 0),
        "length_discipline_rate": _rate(rows, lambda r: r.within_length_band),
    }
    per_stratum = _per_stratum(rows)
    pathlib.Path("results/overall.json").write_text(json.dumps(overall, indent=2))
    pathlib.Path("results/per_stratum.json").write_text(json.dumps(per_stratum, indent=2))
    _render_release_decision(overall, per_stratum)

def _rate(rows, pred) -> dict:
    k = sum(1 for r in rows if pred(r))
    n = len(rows)
    return {"rate": k / max(1, n), "ci95": list(wilson_ci(k, n)), "k": k, "n": n}

def _per_stratum(rows: list[ScoredRow]) -> dict:
    by_lang = _group_rate(rows, "lang")
    by_len  = _group_rate(rows, "length_bin")
    by_iss  = _group_rate(rows, "issue_type")
    by_res  = _group_rate(rows, "resolution")
    return {"lang": by_lang, "length_bin": by_len,
            "issue_type": by_iss, "resolution": by_res}

def _group_rate(rows: list[ScoredRow], key: str) -> dict:
    groups: dict[str, list[ScoredRow]] = defaultdict(list)
    for r in rows:
        groups[r.stratum[key]].append(r)
    return {g: _rate(rs, lambda r: r.acceptable) for g, rs in groups.items()}
```

Design notes on the harness:

- **Separate metric calculation from release decision.** `_emit_reports` writes raw numbers. `_render_release_decision` (not shown; a 40-line function) reads thresholds from `EVAL_SPEC.md` and emits a pass/fail/comment-bundle. Keeping them separate means changing the gate thresholds does not require reruns.
- **Every metric has a Wilson 95% CI.** Wilson score is the correct interval for a proportion; a naive normal-approximation interval is wrong near 0 and 1 (acceptable-rate near 70% is far enough from the edge, but PII leakage near 0% is not).
- **Hallucination and PII are deterministic checks.** They do not depend on a judge or a rubric; they are the "red lines" that fail release regardless of the aggregate score.

### 2.5 Data file format

```jsonl
{"doc_id": "a1b2c3d4e5f60718", "lang": "en", "stratum": {"lang": "en", "length_bin": "medium", "issue_type": "billing", "resolution": "resolved"}, "turns": [{"role": "customer", "text": "..."}, ...], "created_at": "2026-10-14T10:22:01Z", "ground_truth_summary": "...", "notes": "double-annotated"}
```

Required fields: `doc_id`, `stratum`, `turns`, `created_at`, `ground_truth_summary`. `notes` flags the 400 double-annotated examples used for the judge calibration (exercise 03) and the human-vs-judge κ.

### 2.6 Directory layout (as delivered)

```
outputs/exercise-02/
├── EVAL_SPEC.md
├── build_eval.py
├── score.py
├── power_analysis.md
├── data/
│   ├── eval_v1.jsonl                   # frozen; checked into DVC / artifact store
│   └── manifest.json                   # stratum counts, shortages, freeze date
├── results/
│   ├── overall.json
│   ├── per_stratum.json
│   └── release_decision.md
└── README.md                           # how to run score.py against a fresh model output
```

### 2.7 Adapting to other task types

| Task shape | Primary metric | Red-line metric | Reference artifact |
|---|---|---|---|
| **Classification** (intent, toxicity, routing) | Macro-F1 across classes; worst-class recall for safety-critical classes | False-negative rate on the safety-critical class | Human-labeled multi-annotator set with κ |
| **Extraction** (entity/key-value/JSON schema) | Field-level F1; structural-validity rate (parses against the schema) | JSON-parse failure rate; hallucinated-field rate | Ground-truth annotations keyed by doc_id |
| **Open-ended generation** (summarization, draft replies) | Rubric mean + acceptable-rate (this template) | Hallucination rate; PII leakage | Human reference summary + rubric |
| **Tool use** (function calling) | Task success rate on an evaluator loop; argument-validity rate | Unauthorized-tool-call rate | A simulated tool environment with assertions |
| **Multi-turn dialog** | Per-turn rubric + task-success at end; coherence rate | Contradiction rate across turns | Scripted dialog trees with expected end-states |

In all cases, three parts of the template survive: **the spec**, **the stratified split**, and **the power analysis**. The scoring function and the red-line metrics are task-specific.

## 3. Validation steps

1. **Spec is complete and dated.** `EVAL_SPEC.md` names task, population, partition, primary metric, secondary metrics, pass thresholds, strata, sample size, and freeze date. A missing section invalidates the eval; the spec is the artifact, not the numbers.
2. **Time-partitioned split.** `build_eval.py`'s `_train_eval_split_by_time` draws from a date range that does not overlap the training-data date range. Grep `created_at` in `data/eval_v1.jsonl` and confirm every row falls in the specified eval window.
3. **Stratum coverage matches the spec.** `manifest.json`'s `by_stratum` has every (lang × length × issue × resolution) cell with ≥ 1 example or is listed in `shortages`. An unmentioned empty cell is a hard fail.
4. **Deterministic re-sampling.** Running `build_eval.py` twice against the same pool produces byte-identical `eval_v1.jsonl`. If not, the sampling is non-deterministic and results will drift.
5. **Power analysis supports the thresholds.** The release gate in §2.1.6 ("no regression > 5%") is consistent with the minimum detectable effect at n=1200 (≈ 3.7% absolute). If the spec claimed "detect 1% regressions" at n=1200, the power analysis would catch it.
6. **Red-line metrics defined and measurable.** `score.py` produces `hallucination_rate` and `pii_leakage_rate` from deterministic checks, not from a judge. These two must be non-judge metrics because a judge failure is a plausible cause of a leaked PII item going undetected.
7. **Rubric grader provenance is on every row.** Each `ScoredRow` has `rubric_grader ∈ {human, judge}`. A reader of `overall.json` can tell what fraction of the headline number came from humans vs. an LLM judge (exercise 03).
8. **Metadata captures contamination posture.** The spec cross-references exercise 04. The manifest records the training-data date range and the eval date range, which exercise 04 will consume for the hash-level contamination check.
9. **Dataset version is frozen in the filename and manifest.** `eval_v1.jsonl` + `manifest.json` with `version: "1.0"`. A v2 is a *new file*; existing results do not carry over.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Spec completeness** | Missing one of: task, population, partition, primary metric, pass threshold | All seven sections present, but threshold is not operational ("be accurate") | All sections present, every threshold is a numeric gate with 95% CI and the spec is dated and version-stamped |
| **Partition integrity** | Train/eval overlap is possible (sampled from the same pool) | Partition exists but is not enforced in code / infra | Time-based or source-based partition enforced in `build_eval.py`; eval bucket has ACLs that forbid the training pipeline's service account; date ranges documented |
| **Stratification** | Flat random sample | Stratified by one dimension | Stratified by ≥ 3 dimensions matching product-relevant slices (language, length, task-type, outcome); cell shortages recorded in the manifest |
| **Power analysis** | Not attempted | n chosen by rule of thumb; no MDE | Sample-size math written, MDE computed, release gate *consistent* with MDE; per-stratum power limitations documented |
| **Harness quality** | Script computes point estimates only | CIs on headline metric only | Wilson CI on every rate; deterministic red-line checks (hallucination, PII) separate from the judge; `rubric_grader` provenance on every row; per-stratum breakdown emitted automatically |

**Pass threshold:** 7/10, with no single dimension at 0 and the partition-integrity dimension at 2.

## 5. Common mistakes

- **Scraping an eval set from the same pool as training.** The most common failure mode of exercise 02. Even if `random_state=42` splits cleanly, records that are near-duplicates (e.g. the same conversation routed through two channels) leak across the split. Time- or source-based partition is the only robust defense; hash-level dedup (exercise 04) is a backstop, not the primary.
- **Treating the eval set as "evaluation data" the team can edit freely.** Edits mid-cycle invalidate every prior result. Version the dataset; "eval_v2" is a new artifact.
- **Writing a rubric that includes "is the answer good" as a single dimension.** The rubric must be *decomposable*: accuracy, completeness, hallucination, length, policy compliance — each gradable independently. A single-dimension rubric cannot tell you which axis the model regressed on.
- **Picking n before the power analysis.** 1,000 is a round number, not a sample-size calculation. The correct order is: define the smallest effect that matters (δ), define α and β, solve for n. If budget forbids the computed n, document the resulting MDE and gate accordingly.
- **Uniform per-cell quota under skewed traffic.** If prod is 95% English, 1200 eval examples at 50/50 en/es produces a Spanish-heavy eval that overstates Spanish performance in the overall metric. Weight cell quotas by traffic *and* report per-cell separately.
- **Using ROUGE-L as the primary metric for generation.** ROUGE measures n-gram overlap against one reference; models can be rewarded for copying the reference's phrasing and punished for correct paraphrases. Keep ROUGE as a secondary tripwire; the primary is a rubric score with human (or calibrated-judge) grading.
- **Skipping the red-line metrics.** A model with a 0.5% PII leakage rate can pass the rubric gate while being unreleasable. Deterministic checks for PII and hallucination are non-negotiable and must *fail* the release decision independently of the aggregate.
- **No manifest.** The `data/eval_v1.jsonl` file alone is not enough; the next person on the team needs the manifest (`version`, `frozen_on`, `seed`, `by_stratum`, `shortages`) to know what the file is and whether a run against it is comparable.
- **Reporting only the aggregate.** "Mean rubric 3.9" hides an `es + long + abuse` cell at 2.4. Per-stratum metrics must be emitted by the harness and read by the reviewer; the release gate can allow weaker per-cell numbers but a *missing* per-cell report is disqualifying.
- **Confusing "held out from training" with "held out from training-data-construction code."** If the data pipeline's filtering script was tuned on the eval set, the eval set is contaminated with selection bias even if no row appears in training. The spec names which code and which humans have seen the eval set.

## 6. References

- NIST AI Risk Management Framework — *Measure 2.3* (performance in deployment-relevant conditions) and *Measure 2.7* (safety of AI systems including unintended harms) are the direct standards for a stratified, red-line-gated held-out eval. <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF Generative AI Profile — expands the Measure function with specific guidance on holdout eval design for generative systems. <https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/>
- OWASP Machine Learning Security Top 10 — *ML02 Data Poisoning* and *ML04 Membership Inference Attack* are both mitigated by a time-partitioned, access-controlled held-out eval; *ML06 AI Supply Chain Vulnerabilities* applies to pinning the dataset version and manifest. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — tactics catalog; the training/eval partition is a direct defense against *ML Supply Chain Compromise* and *Evade ML Model* categories where an adversary could craft inputs similar to training data to game the eval. <https://atlas.mitre.org/>
- Wilson (1927) — binomial score interval; used in `wilson_ci`. <https://www.jstor.org/stable/2276774>
- Hugging Face Datasets — `load_dataset` and the dataset-card specification that `data/manifest.json` mirrors for an internal eval. <https://huggingface.co/docs/datasets/>
