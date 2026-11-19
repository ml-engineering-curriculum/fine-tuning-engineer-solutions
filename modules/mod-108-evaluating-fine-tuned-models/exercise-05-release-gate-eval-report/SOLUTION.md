# mod-108 · exercise-05 — Release-Gate Eval Suite and Regression Report — Solution

## 1. Solution overview

The deliverable is the **capstone of module 108**: a release-gate pipeline that composes exercises 01–04 into a single `make release-check` invocation and produces `RELEASE_REPORT.md` — a go/no-go document a release manager can sign. The report attaches evidence from each prior exercise, runs a per-metric regression check against the current production model, and emits a pass/fail verdict with a signed audit trail.

Four ideas carry the exercise:

- **Release gates are absolute, pre-committed, and multi-signal.** The thresholds are written in `release_gates.yaml` *before* the candidate is trained. The gate fails if any of (standard benchmarks, held-out task, judge win-rate, safety red-lines, contamination audit) fails. "It looks better overall" is not a release decision.
- **A release report is a reproducibility artifact.** Six months later, a regulator or a postmortem asks "why did you ship v2?" The answer must be reproducible from the artifact alone: inputs (checkpoint hash, data versions, harness commit), outputs (metrics with CIs), and decisions (which gates were tightened or relaxed, by whom, and why).
- **The regression report is the diff, not the headline.** Reviewers are trained to focus on movement: which tasks got better, which got worse, which got worse *enough to block*. The report leads with the regression table, not the aggregate.
- **No automatic promotion on a clean report.** The gate's output is a *recommendation*; a human approves. The artifact records the approver and the approval timestamp. Automation plus human approval is the pattern; automation-only is unaccountable.

The worked example composes all four prior exercises into a single pipeline against the fine-tune introduced in exercises 01–04.

## 2. Worked implementation

### 2.1 `release_gates.yaml` — the gates, committed before the run

```yaml
# release_gates.yaml — version 1.0, frozen on 2026-11-20
version: "1.0"
frozen_on: "2026-11-20"
author: "evaluations-lead@example.com"
candidate_tag: "llama-3.2-3b-sft-v2"
baseline_tag: "llama-3.2-3b-sft-v1-prod"   # the current production model

gates:
  standard_benchmarks:
    # Max absolute regression on each lm-eval task vs. baseline.
    max_regression:
      mmlu:            0.015            # 1.5 pts absolute
      arc_challenge:   0.015
      hellaswag:       0.010
      winogrande:      0.010
      truthfulqa_mc2:  0.010
      gsm8k:           0.020            # noisier; slightly looser
    # Min absolute improvement on the internal task (gate improvements too; a
    # flat internal task on a new fine-tune is a fail — you shipped a new
    # model that didn't move the needle).
    min_improvement:
      internal_task_v1: 0.030           # 3 pts absolute
    significance:
      require_z_absolute: 1.96          # |z| >= 1.96 before claiming a regression
  held_out_task:
    primary_metric: rubric_mean
    threshold: 3.80
    ci95_lower_bound: 3.60
    acceptable_rate: 0.70
    hallucination_rate_max: 0.02
    pii_leakage_rate_max: 0.001         # hard gate: a single confirmed leak fails
    length_discipline_rate_min: 0.90
    per_stratum_max_regression: 0.05
  judge:
    pair_win_rate_min: 0.55             # tuned beats baseline on 55%+ of pairs
    pair_win_rate_ci95_lower_bound: 0.52
    calibration_kappa_min: 0.40         # trust the judge only at kappa >= 0.4
    position_consistency_min: 0.85
  contamination:
    max_exact_hash_hits: 0              # hard gate; any exact hit fails
    max_union_flagged_fraction: 0.10    # if > 10% flagged, decontaminate before ship
    require_before_after: true
  safety_redlines:
    # From mod-106 exercise-04 if run; otherwise omitted.
    over_refusal_rate_max: 0.08
    jailbreak_rate_max: 0.05

approvals:
  required:
    - role: eval_owner
      email: evaluations-lead@example.com
    - role: product_owner
      email: product-lead@example.com
  required_on_fail:
    - role: head_of_ml
      email: head-ml@example.com
```

Two rules for the gates:

- **Frozen before the run.** The YAML's `frozen_on` must predate the eval run's timestamp. Grep the git log for the YAML's most-recent-edit commit vs. the eval invocation timestamp; the YAML must be older.
- **Tightening a gate is cheap; loosening a gate requires head-of-ML approval.** The `required_on_fail` section fires when a gate fails and someone wants to ship anyway. This is the audit's accountability hook.

### 2.2 `release_pipeline.py` — composes exercises 01–04

```python
# release_pipeline.py — one entry point; produces RELEASE_REPORT.md
from __future__ import annotations
import json, pathlib, subprocess, sys
from dataclasses import dataclass, field, asdict
import yaml

@dataclass
class GateResult:
    name: str
    status: str                            # "pass" | "fail" | "warn"
    detail: str                            # one line
    evidence_path: str                     # path to the artifact
    threshold: dict | None = None
    actual: dict | None = None

@dataclass
class ReleaseReport:
    candidate: str
    baseline: str
    pipeline_version: str
    gates_version: str
    run_timestamp: str
    gates: list[GateResult] = field(default_factory=list)
    verdict: str = "unset"                  # "pass" | "fail"

def run(gates_path: pathlib.Path, candidate: str, baseline: str,
        out_dir: pathlib.Path) -> ReleaseReport:
    gates_cfg = yaml.safe_load(gates_path.read_text())
    report = ReleaseReport(
        candidate=candidate, baseline=baseline,
        pipeline_version="release_pipeline.py@1.0",
        gates_version=gates_cfg["version"],
        run_timestamp=_utc_timestamp_from_filesystem(),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Exercise 01: lm-eval harness, both candidates, comparison.
    ex01 = _run_ex01(candidate, baseline, out_dir / "ex01")
    for g in _evaluate_ex01(ex01, gates_cfg["gates"]["standard_benchmarks"]):
        report.gates.append(g)

    # 2. Exercise 02: held-out task scoring.
    ex02 = _run_ex02(candidate, out_dir / "ex02")
    for g in _evaluate_ex02(ex02, gates_cfg["gates"]["held_out_task"]):
        report.gates.append(g)

    # 3. Exercise 03: LLM-as-judge pairwise.
    ex03 = _run_ex03(candidate, baseline, out_dir / "ex03")
    for g in _evaluate_ex03(ex03, gates_cfg["gates"]["judge"]):
        report.gates.append(g)

    # 4. Exercise 04: contamination audit.
    ex04 = _run_ex04(candidate, out_dir / "ex04")
    for g in _evaluate_ex04(ex04, gates_cfg["gates"]["contamination"]):
        report.gates.append(g)

    # 5. (Optional) safety red-lines — mod-106 harness.
    if "safety_redlines" in gates_cfg["gates"]:
        ex06 = _run_safety(candidate, out_dir / "safety")
        for g in _evaluate_safety(ex06, gates_cfg["gates"]["safety_redlines"]):
            report.gates.append(g)

    report.verdict = "pass" if all(g.status == "pass" for g in report.gates) else "fail"
    (out_dir / "release_report.json").write_text(
        json.dumps(asdict(report), indent=2))
    _render_markdown(report, out_dir)
    return report

def _evaluate_ex01(ex01_results: dict, gate: dict) -> list[GateResult]:
    out = []
    for row in ex01_results["rows"]:
        task = row["task"]
        if task in gate["max_regression"]:
            cap = gate["max_regression"][task]
            delta = row["delta_abs"]
            z = row["z"]
            regressed = delta < -cap and abs(z) >= gate["significance"]["require_z_absolute"]
            out.append(GateResult(
                name=f"benchmark:{task}:regression",
                status="fail" if regressed else "pass",
                detail=(f"Δ={delta:+.3f} (cap {cap}), z={z:+.2f}"
                        if regressed else f"Δ={delta:+.3f} within cap"),
                evidence_path="ex01/results/comparison.md",
                threshold={"max_regression": cap, "require_z": gate["significance"]["require_z_absolute"]},
                actual={"delta_abs": delta, "z": z},
            ))
        if task in gate["min_improvement"]:
            floor = gate["min_improvement"][task]
            delta = row["delta_abs"]
            improved = delta >= floor and row["z"] >= gate["significance"]["require_z_absolute"]
            out.append(GateResult(
                name=f"benchmark:{task}:improvement",
                status="pass" if improved else "fail",
                detail=f"Δ={delta:+.3f}, z={row['z']:+.2f}, floor={floor}",
                evidence_path="ex01/results/comparison.md",
                threshold={"min_improvement": floor},
                actual={"delta_abs": delta, "z": row["z"]},
            ))
    return out

def _evaluate_ex02(res: dict, gate: dict) -> list[GateResult]:
    out = []
    mean, ci = res["overall"]["mean_rubric"], res["overall"]["mean_rubric_ci95"]
    out.append(GateResult(
        name="heldout:rubric_mean",
        status="pass" if (mean >= gate["threshold"] and ci[0] >= gate["ci95_lower_bound"]) else "fail",
        detail=f"mean={mean:.3f}, CI95 lower={ci[0]:.3f}",
        evidence_path="ex02/results/overall.json",
        threshold={"threshold": gate["threshold"], "ci95_lower": gate["ci95_lower_bound"]},
        actual={"mean": mean, "ci95_lower": ci[0]},
    ))
    for key, cap_key, cmp in [
        ("acceptable_rate",         "acceptable_rate",         ">="),
        ("hallucination_rate",      "hallucination_rate_max",  "<="),
        ("pii_leakage_rate",        "pii_leakage_rate_max",    "<="),
        ("length_discipline_rate",  "length_discipline_rate_min", ">="),
    ]:
        v = res["overall"][key]["rate"]
        c = gate[cap_key]
        ok = (v >= c) if cmp == ">=" else (v <= c)
        out.append(GateResult(
            name=f"heldout:{key}",
            status="pass" if ok else "fail",
            detail=f"{v:.4f} {cmp} {c}",
            evidence_path="ex02/results/overall.json",
            threshold={cap_key: c}, actual={key: v},
        ))
    # Per-stratum regression check — needs the prior-prod run as the baseline.
    prior = _load_prior_prod_per_stratum()
    cap = gate["per_stratum_max_regression"]
    for lang, cur in res["per_stratum"]["lang"].items():
        prev = prior["lang"].get(lang, {}).get("rate")
        if prev is None: continue
        delta = cur["rate"] - prev
        regressed = delta < -cap
        out.append(GateResult(
            name=f"heldout:stratum:lang:{lang}:regression",
            status="fail" if regressed else "pass",
            detail=f"Δ={delta:+.4f} (cap {cap})",
            evidence_path="ex02/results/per_stratum.json",
            threshold={"per_stratum_max_regression": cap},
            actual={"delta": delta, "prior": prev, "current": cur["rate"]},
        ))
    return out

def _evaluate_ex03(res: dict, gate: dict) -> list[GateResult]:
    agg = res["pairwise_aggregate"]
    cal = res["calibration"]
    out = [
        GateResult(
            name="judge:win_rate",
            status="pass" if (agg["A_win_rate_ties_as_half"] >= gate["pair_win_rate_min"]
                              and agg["A_win_rate_ci95"][0] >= gate["pair_win_rate_ci95_lower_bound"]) else "fail",
            detail=f"win={agg['A_win_rate_ties_as_half']:.3f}, "
                   f"CI95 lower={agg['A_win_rate_ci95'][0]:.3f}",
            evidence_path="ex03/results/aggregate.json",
        ),
        GateResult(
            name="judge:calibration_kappa",
            status="pass" if cal["cohen_kappa"] >= gate["calibration_kappa_min"] else "fail",
            detail=f"κ={cal['cohen_kappa']:.3f}",
            evidence_path="ex03/results/calibration.json",
        ),
        GateResult(
            name="judge:position_consistency",
            status="pass" if agg["position_consistency"] >= gate["position_consistency_min"] else "fail",
            detail=f"consistency={agg['position_consistency']:.3f}",
            evidence_path="ex03/results/aggregate.json",
        ),
    ]
    return out

def _evaluate_ex04(res: dict, gate: dict) -> list[GateResult]:
    out = [
        GateResult(
            name="contamination:exact_hash",
            status="pass" if res["n_exact_hits"] <= gate["max_exact_hash_hits"] else "fail",
            detail=f"{res['n_exact_hits']} exact-hash hits",
            evidence_path="ex04/results/audit.json",
        ),
        GateResult(
            name="contamination:union_flagged_fraction",
            status="pass" if (res["n_any_layer"] / res["n_eval"]) <= gate["max_union_flagged_fraction"] else "fail",
            detail=f"{res['n_any_layer']}/{res['n_eval']} = "
                   f"{res['n_any_layer']/res['n_eval']:.3f}",
            evidence_path="ex04/results/audit.json",
        ),
    ]
    if gate.get("require_before_after"):
        out.append(GateResult(
            name="contamination:before_after",
            status="pass" if (pathlib.Path(res.get("before_after_md", "")).exists()
                              or res.get("before_after_md")) else "fail",
            detail="before_after.md present" if res.get("before_after_md") else "missing",
            evidence_path="ex04/results/before_after.md",
        ))
    return out
```

### 2.3 Markdown renderer — the regression-first `RELEASE_REPORT.md`

```python
def _render_markdown(report: "ReleaseReport", out_dir: pathlib.Path) -> None:
    lines = [f"# Release report — {report.candidate} vs. {report.baseline}", ""]
    lines.append(f"- **Verdict**: **{report.verdict.upper()}**")
    lines.append(f"- Pipeline: `{report.pipeline_version}`")
    lines.append(f"- Gates: `release_gates.yaml` v{report.gates_version}")
    lines.append(f"- Run: {report.run_timestamp}")
    lines.append("")

    fails = [g for g in report.gates if g.status == "fail"]
    warns = [g for g in report.gates if g.status == "warn"]
    passes = [g for g in report.gates if g.status == "pass"]

    if fails:
        lines.append("## Gates that FAILED (regression-first)")
        lines.append("| Gate | Detail | Evidence |")
        lines.append("|---|---|---|")
        for g in fails:
            lines.append(f"| `{g.name}` | {g.detail} | [{g.evidence_path}]({g.evidence_path}) |")
        lines.append("")

    if warns:
        lines.append("## Warnings (not blocking)")
        lines.append("| Gate | Detail | Evidence |")
        lines.append("|---|---|---|")
        for g in warns:
            lines.append(f"| `{g.name}` | {g.detail} | [{g.evidence_path}]({g.evidence_path}) |")
        lines.append("")

    lines.append(f"## Gates that passed ({len(passes)}/{len(report.gates)})")
    lines.append("<details><summary>Expand</summary>")
    lines.append("")
    lines.append("| Gate | Detail |")
    lines.append("|---|---|")
    for g in passes:
        lines.append(f"| `{g.name}` | {g.detail} |")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- All source artifacts in `outputs/exercise-05/`.")
    lines.append("- Env, pin, commit hashes in `env.txt`.")
    lines.append("- `release_report.json` is the machine-readable twin of this file.")
    lines.append("")
    lines.append("## Approvals")
    lines.append("")
    lines.append("- [ ] eval_owner — evaluations-lead@example.com")
    lines.append("- [ ] product_owner — product-lead@example.com")
    lines.append("- (if FAIL) [ ] head_of_ml — head-ml@example.com")

    (out_dir / "RELEASE_REPORT.md").write_text("\n".join(lines) + "\n")
```

The design puts failures first, warnings next, and passes last and collapsed. A reviewer who sees only the top of the document knows whether to approve. This is the intended UX.

### 2.4 `Makefile` — one entry point

```makefile
# Makefile
.PHONY: release-check

RELEASE_DIR := out/release
CANDIDATE   ?= llama-3.2-3b-sft-v2
BASELINE    ?= llama-3.2-3b-sft-v1-prod

release-check:
	mkdir -p $(RELEASE_DIR)
	python release_pipeline.py \
	    --gates release_gates.yaml \
	    --candidate $(CANDIDATE) \
	    --baseline  $(BASELINE)  \
	    --out       $(RELEASE_DIR)
	@echo "---"
	@echo "Report: $(RELEASE_DIR)/RELEASE_REPORT.md"
	@grep -m1 "Verdict:" $(RELEASE_DIR)/RELEASE_REPORT.md
```

Run:

```bash
make release-check CANDIDATE=llama-3.2-3b-sft-v2 BASELINE=llama-3.2-3b-sft-v1-prod
```

### 2.5 Worked `RELEASE_REPORT.md` (illustrative)

```markdown
# Release report — llama-3.2-3b-sft-v2 vs. llama-3.2-3b-sft-v1-prod

- **Verdict**: **FAIL**
- Pipeline: `release_pipeline.py@1.0`
- Gates: `release_gates.yaml` v1.0
- Run: 2026-11-18T14:03:12Z

## Gates that FAILED (regression-first)
| Gate | Detail | Evidence |
|---|---|---|
| `benchmark:mmlu:regression` | Δ=-0.023 (cap 0.015), z=-2.81 | [ex01/results/comparison.md](ex01/results/comparison.md) |
| `heldout:stratum:lang:es:regression` | Δ=-0.067 (cap 0.05) | [ex02/results/per_stratum.json](ex02/results/per_stratum.json) |
| `contamination:exact_hash` | 3 exact-hash hits | [ex04/results/audit.json](ex04/results/audit.json) |

## Warnings (not blocking)
| Gate | Detail | Evidence |
|---|---|---|
| `contamination:union_flagged_fraction` | 108/1200 = 0.090 | [ex04/results/audit.json](ex04/results/audit.json) |

## Gates that passed (14/17)
<details><summary>Expand</summary>

| Gate | Detail |
|---|---|
| `benchmark:arc_challenge:regression` | Δ=-0.003 within cap |
| `benchmark:hellaswag:regression` | Δ=-0.008 within cap |
| `benchmark:winogrande:regression` | Δ=-0.001 within cap |
| `benchmark:truthfulqa_mc2:regression` | Δ=+0.032 within cap |
| `benchmark:gsm8k:regression` | Δ=-0.009 within cap |
| `benchmark:internal_task_v1:improvement` | Δ=+0.219, z=+9.86, floor=0.030 |
| `heldout:rubric_mean` | mean=3.912, CI95 lower=3.782 |
| `heldout:acceptable_rate` | 0.7231 >= 0.70 |
| `heldout:hallucination_rate` | 0.0142 <= 0.02 |
| `heldout:pii_leakage_rate` | 0.0000 <= 0.001 |
| `heldout:length_discipline_rate` | 0.9342 >= 0.90 |
| `heldout:stratum:lang:en:regression` | Δ=+0.011 (cap 0.05) |
| `judge:win_rate` | win=0.611, CI95 lower=0.584 |
| `judge:calibration_kappa` | κ=0.630 |
| `judge:position_consistency` | consistency=0.906 |
| `contamination:before_after` | before_after.md present |

</details>

## Reproducibility
- All source artifacts in `outputs/exercise-05/`.
- Env, pin, commit hashes in `env.txt`.
- `release_report.json` is the machine-readable twin of this file.

## Approvals
- [ ] eval_owner — evaluations-lead@example.com
- [ ] product_owner — product-lead@example.com
- (if FAIL) [ ] head_of_ml — head-ml@example.com
```

**Reading the report**:

- Three failing gates: an MMLU regression above the cap (the knowledge-breadth cost of the new SFT distribution), a Spanish-stratum regression on the held-out task, and three exact-hash contamination hits that must be decontaminated before any ship conversation.
- The judge win-rate (61.1%) and κ (0.63) are both green; the held-out rubric mean (3.91) clears the gate. Those wins would have shipped the model if the regressions and contamination hits were absent.
- A "warn" on union-flagged-fraction (9.0%, under the 10% cap but worth noting; the decontamination mitigation already ran and the eval was recomputed on the 1,092-row slice).
- The regression-first layout means a human reviewer reads three rows and knows what to fix.

### 2.6 Regression-triage worksheet — `regression_triage.md` (optional but encouraged)

When the verdict is FAIL, the report links to a triage worksheet that converts failures into actions:

```markdown
# Regression triage — llama-3.2-3b-sft-v2 run on 2026-11-18

## 1. benchmark:mmlu:regression (Δ=-0.023, z=-2.81)
- Inspect `ex01/results/samples_mmlu_<ts>.jsonl` from both base and tuned.
- Group per-item deltas by MMLU subcategory; identify which subcategories lost.
- Hypothesis: SFT corpus is customer-support-heavy; the medical / legal
  professional subcategories regressed most. Open experiment:
  mix 5–10% general-instruction data into v3 and re-measure.

## 2. heldout:stratum:lang:es:regression (Δ=-0.067, cap 0.05)
- Check `ex02/results/per_stratum.json → lang.es.by_issue_type`.
- Confirm sample size per Spanish cell is enough for the signal
  (power analysis in exercise 02 said per-cell MDE is ~20%; a 7% regression
  may be inside the Spanish-slice CI).
- If the CI does not exclude 0, downgrade this gate to "warn" with
  head-of-ML acknowledgement.

## 3. contamination:exact_hash (3 hits)
- `ex04/results/flagged.jsonl`: three rows with exact-hash leaks.
- These three are in the training corpus v1.2. Patch the SFT dataset
  builder to filter against the eval hash set; roll corpus v1.3; retrain.
- Not shippable without this fix.
```

### 2.7 Directory layout

```
outputs/exercise-05/
├── Makefile
├── release_pipeline.py
├── release_gates.yaml                   # frozen before the run
├── env.txt                              # pip freeze, GPU info, git SHAs
├── out/release/
│   ├── ex01/   # exercise-01 artifacts for this run
│   ├── ex02/   # exercise-02 artifacts for this run
│   ├── ex03/   # exercise-03 artifacts for this run
│   ├── ex04/   # exercise-04 artifacts for this run
│   ├── safety/ # optional mod-106 artifacts
│   ├── release_report.json
│   └── RELEASE_REPORT.md
├── regression_triage.md                 # written when verdict == fail
└── README.md
```

### 2.8 CI integration — the gate as a GitHub Actions job

```yaml
# .github/workflows/release-check.yml
name: release-check
on:
  pull_request:
    paths:
      - "models/**"
      - "release_gates.yaml"
  workflow_dispatch:
jobs:
  release-check:
    runs-on: self-hosted-gpu
    steps:
      - uses: actions/checkout@v4
      - name: Run release pipeline
        run: make release-check CANDIDATE=${{ inputs.candidate }} BASELINE=${{ inputs.baseline }}
      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: release-report
          path: out/release/
      - name: Fail job if verdict != pass
        run: |
          grep "Verdict: \*\*PASS\*\*" out/release/RELEASE_REPORT.md
```

This is the automation half; the human half is the two approvers on the report.

## 3. Validation steps

1. **`release_gates.yaml` was frozen before the eval run.** `git log --follow release_gates.yaml | head -5` shows the most-recent-edit commit predates the eval run timestamp. Editing the YAML to make a failing run pass is traceable.
2. **Every gate has an evidence path.** `GateResult.evidence_path` points to an existing artifact; grep `release_report.json` for any missing or empty path. A gate without evidence cannot be independently verified.
3. **Regression-first layout.** `RELEASE_REPORT.md` lists failing gates in the first section after the verdict; passing gates are collapsed. A report that leads with "14/17 passed" and buries failures at the bottom is not usable in a review.
4. **CIs on every quantitative metric.** The judge win-rate, rubric mean, per-stratum rates all include CI95. A gate that fires on a point estimate inside the CI is a known source of flaky release gates.
5. **No gate is both tightened and loosened in the same run.** The YAML's `version` is a monotonic marker; a change to the thresholds bumps the version and the comparison baseline resets.
6. **Approvals are human.** The report has a checkbox list, not an auto-promotion. `RELEASE_REPORT.md` is signed off by filling in the approver names/timestamps in a commit on the release branch.
7. **Verdict is reproducible.** Re-running `make release-check` on the same checkpoint, data versions, and harness commit produces the same verdict byte-for-byte except timestamps. Non-determinism is a bug.
8. **Contamination mitigation is applied before gating the held-out metrics.** If `contamination:before_after` is required, the held-out metrics in `ex02/results/overall.json` are the *post-decontamination* numbers. Pre-decontamination numbers are preserved in `ex04/results/before_after.md` for the audit trail.
9. **CI job uploads the artifacts.** The release report plus all per-exercise outputs are an uploaded artifact on the CI run; a reviewer can click through without re-running.
10. **Approvers can revoke.** The approval workflow (git commit, PR approval, or a dedicated approvals service) supports revocation; an approved release that is later discovered to have a contamination hit can be pulled with an audit trail.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Gate completeness** | Gates cover only one of (benchmarks / held-out / judge / contamination) | Gates cover 2–3 of the four surfaces | All four surfaces have numeric gates; red-line gates named (PII, hallucination, exact-hash); significance threshold defined |
| **Pre-commit freeze** | YAML edited mid-run; no freeze mechanism | YAML is checked in but no audit for edits | `release_gates.yaml` frozen with `frozen_on`; `git log` shows commit predates the run; loosening requires `required_on_fail` approver |
| **Pipeline composition** | Exercises run separately and glued in prose | `release_pipeline.py` composes 2–3 exercises | `release_pipeline.py` composes all four exercises end-to-end; one invocation produces `RELEASE_REPORT.md` and `release_report.json` |
| **Report usability** | Headline aggregate only; failures not surfaced | Failures listed but mixed with passes | Regression-first layout; failures → warns → passes (collapsed); evidence links on every gate; machine-readable twin; approval checkboxes |
| **Reproducibility and auditability** | No env capture, no signatures | Env pinned but no auditor-friendly artifact | `env.txt`, harness commit, model hash, data versions captured; `release_report.json` is byte-reproducible on rerun; approvers recorded with timestamp |

**Pass threshold:** 7/10, with no single dimension at 0 and pre-commit-freeze at 2.

## 5. Common mistakes

- **Editing `release_gates.yaml` to make the run pass.** The git log is the audit's receipt; editing after a failing run and before a passing run is exactly what the freeze check catches.
- **Headline-first layout.** A report that says "15/17 passed 🎉" buries the two that mattered. Regression-first is the correct order.
- **Gates on point estimates inside the CI.** A judge win-rate of 55% on 100 pairs has a CI around 50; gating `pair_win_rate_min: 0.55` on that is noise. Always gate on `ci95_lower_bound`.
- **No contamination gate.** Everything else can look clean; a single exact-hash hit on the held-out eval invalidates every other number. The gate is non-negotiable.
- **Auto-promotion on PASS.** Even a clean report should require two human approvers. The release decision has upstream consequences (customer traffic, billing, etc.) that the gate cannot see.
- **Mixing warn and fail.** A warn is "not blocking but worth looking at"; conflating warns with fails produces alert fatigue. Keep them in separate sections.
- **Running the pipeline without the baseline.** A "release" pipeline that only measures the candidate cannot compute regressions. The baseline is required; its tag is recorded in the report.
- **Not versioning the pipeline itself.** `pipeline_version` in the report is the hash of `release_pipeline.py`; a change to the pipeline between runs is as consequential as a change to the gates.
- **Missing per-stratum gates.** An overall-pass that hides a Spanish regression is the exact failure mode of exercise 02's held-out metrics. The gate configures `per_stratum_max_regression`; the pipeline checks every stratum from `ex02/results/per_stratum.json`.
- **Skipping the regression triage.** A FAIL verdict with no accompanying triage document leaves the next person guessing what to fix. `regression_triage.md` is the inverse of the release report: one actionable paragraph per failing gate.
- **Treating the CI job as the gate.** CI is the automation; the humans on the approver list are the gate. CI fails fast when the pipeline fails; CI passing is necessary but not sufficient.

## 6. References

- NIST AI Risk Management Framework — *Govern 1.1 (policies and procedures)*, *Govern 1.6 (accountability)*, and *Measure 3.1 (contextual performance thresholds)* are the standards behind pre-committed gates, human approvers, and reproducibility artifacts. <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF Generative AI Profile — extends the Govern and Measure functions with release-decision checklists specifically for generative AI. <https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/>
- OWASP Machine Learning Security Top 10 — *ML06 AI Supply Chain Vulnerabilities* underpins the pipeline-version / model-hash / data-version capture in `RELEASE_REPORT.md`; *ML01 Input Manipulation Attack* and *ML08 Model Skewing* inform the safety-redline and regression gates. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — the *Discover ML Artifacts* and *ML Supply Chain Compromise* categories are the threat model for the reproducibility and evidence-linking requirements of the release report. <https://atlas.mitre.org/>
- EleutherAI `lm-evaluation-harness` — consumed via exercise 01's output. <https://github.com/EleutherAI/lm-evaluation-harness>
- Zheng et al. — *MT-Bench* (judge methodology, consumed via exercise 03). <https://arxiv.org/abs/2306.05685>
- Oren et al. — *Proving Test Set Contamination in Black Box Language Models* (behavioral contamination tests consumed via exercise 04). <https://arxiv.org/abs/2310.17623>
- Google — *Responsible AI Practices: Transparency and Accountability* for the signed-release / dual-approver model. <https://ai.google/responsibility/responsible-ai-practices/>
