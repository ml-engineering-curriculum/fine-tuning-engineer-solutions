# mod-111 · exercise-02 — Dataset Budget Allocation Across the Objective Stack — Solution

## 1. Solution overview

The deliverable is a **pre-registered allocation of a fixed dataset budget across every objective a post-training stack has to feed**, plus a worked example showing how the allocation was derived and how it will be re-balanced in later iterations. A post-training pipeline eats datasets at six surfaces, and the budget almost never covers all of them at their "ideal" sizes — the exercise is to make the trade-offs explicit, not to pretend they do not exist.

Four ideas carry the exercise:

- **The objective stack is six buckets, not one.** SFT core, SFT long-tail/diversity, preference pairs (DPO/RLHF), safety/refusal data, held-out eval, red-team/adversarial. A plan that lists only "SFT + eval" has silently zeroed four of them.
- **Buckets are coupled, not independent.** Red-team findings feed safety data; preference pairs often reuse SFT prompts with on-policy generations; held-out eval must be decontaminated against *every other* bucket. Allocating as if buckets are independent produces a plan that double-counts annotator time and under-counts decontamination work.
- **Budget units are not just item counts.** Items, token count, annotator-hours, and dollars all cap the plan; any of the four can bind first. A plan that only budgets items but has no annotator-hour forecast will slip the schedule.
- **The plan is iteration-aware.** v1 is a bootstrap allocation; v2 is re-allocated using the v1 evaluation results. The exercise requires both the initial split *and* a re-allocation rule for later releases — NIST AI RMF `Measure-2.7` and `Manage-4.1` both expect this kind of feedback loop to be documented.

The worked example allocates a **35,000-item / 7,000 annotator-hour / $120,000** budget across the six objective buckets for the v1 release of the customer-support-assistant from exercise-01, then shows the v2 re-allocation after the gate run from exercise-03 surfaces where v1 was under-fed.

## 2. Implementation — worked allocation

### 2.1 The six-bucket inventory

`budget_inventory.yaml`:

```yaml
# budget_inventory.yaml — version 1.0
# Freeze this inventory before any annotation work starts. A bucket added or
# removed mid-run invalidates the budget-utilization dashboards downstream.
version: "1.0"
frozen_on: "2026-11-24"
total_budget:
  items: 35000
  annotator_hours: 7000
  dollars_usd: 120000

# Every bucket feeds a downstream objective in the post-training stack.
# Dependencies between buckets are called out explicitly; a bucket that
# is a prerequisite for another cannot be zero without zeroing the dependent.
buckets:

  sft_core:
    purpose: "Teach the base model the exact task shape, chat format, and
              golden-path responses for the customer-support domain."
    feeds:   ["SFT (mod-103)"]
    item_cost_hours: 0.12          # ~7 min / item incl. QA
    quality_floor:
      min_items_for_task: 6000     # below this, under-fitting is the dominant risk
    produces:
      prompts_reusable_for:
        - "preference_pairs (via on-policy generations)"
        - "eval_development (not the held-out set)"

  sft_long_tail:
    purpose: "Cover edge cases, unusual issue types, multi-turn clarifications,
              and the long tail that golden-path templates underfit."
    feeds:   ["SFT (mod-103)"]
    item_cost_hours: 0.25          # ~15 min / item; complex triage rows
    quality_floor:
      min_items_for_long_tail: 3000
    depends_on:
      - "user-research or ticket-sampling pass to surface representative rare cases"

  preference_pairs:
    purpose: "DPO/RLHF signal: chosen vs. rejected, ranked by the reward
              contract the product wants (helpful, format-compliant, safe)."
    feeds:   ["preference optimization (mod-105)"]
    item_cost_hours: 0.33          # ~20 min per pair (two rollouts + ranking)
    quality_floor:
      min_pairs: 3000              # DPO underperforms under this on 7-13B scale
    depends_on:
      - "sft_core prompts (reusable) OR curated preference-prompt distribution"

  safety_and_refusal:
    purpose: "Teach the model to refuse disallowed requests, soft-refuse
              ambiguous requests, and avoid over-refusal on safe content."
    feeds:   ["safety-alignment-tuning (mod-106)"]
    item_cost_hours: 0.30          # ~18 min / item incl. category labeling
    quality_floor:
      min_items_for_refusal_training: 1500
      min_items_for_over_refusal_eval: 500
    depends_on:
      - "policy document naming allowed vs. disallowed vs. sensitive categories"
      - "red_team pass at least once before this bucket exits beta"

  held_out_eval:
    purpose: "Decontaminated held-out eval set for the release-gate pipeline;
              per-stratum coverage of language, issue type, severity."
    feeds:   ["evaluation (mod-108), release gates (exercise-03)"]
    item_cost_hours: 0.40          # ~24 min / item; rubric + gold-answer + stratum tags
    quality_floor:
      min_items: 1200              # below this, per-stratum CIs are too wide
    hard_rule: "contamination-audited vs. every other bucket before freeze"

  red_team:
    purpose: "Adversarial prompts across jailbreak, prompt injection, PII
              extraction, and jailbreak+injection chains."
    feeds:   ["safety-alignment-tuning (mod-106) and release gates"]
    item_cost_hours: 0.45          # ~27 min / item; adversarial crafting + verification
    quality_floor:
      min_items_for_gate: 500
      min_items_for_training_cycle: 1500
    produces:
      findings_into: "safety_and_refusal (converted to refusal training rows)"
```

Three rules make this inventory load-bearing:

- **Dependencies are explicit.** A bucket with `depends_on` cannot start before its upstream is at least at beta size. Red-team is a prerequisite for shipping the safety bucket, not an optional add-on.
- **Quality floors are non-negotiable.** Each bucket has a minimum below which the downstream objective fails. Allocating 1,000 preference pairs because "the budget is tight" produces a DPO run that regresses the model on its own eval.
- **Item cost drives the budget ceiling.** `item_cost_hours × items_in_bucket` is checked against the annotator-hour cap before the plan is accepted. A plan that costs more than 7,000 annotator-hours is not a plan.

### 2.2 v1 allocation — the bootstrap split

`allocation_v1.yaml`:

```yaml
# allocation_v1.yaml — v1.0, frozen 2026-11-24
# Derived from budget_inventory.yaml v1.0.
# This is the bootstrap allocation; v2 will be re-derived from v1 gate results.

allocation:
  sft_core:
    items: 12000
    hours: 1440
    pct_items: 34.3
    note: "Floor + enough coverage of the ten top ticket categories to fine-tune a 7-8B base"

  sft_long_tail:
    items: 4000
    hours: 1000
    pct_items: 11.4
    note: "Floor + ticket-sampling pass produced 12 representative rare categories; ~333 each"

  preference_pairs:
    items: 6000
    hours: 1980
    pct_items: 17.1
    note: "2x the DPO floor; prompts reused from sft_core with on-policy generations"

  safety_and_refusal:
    items: 4000
    hours: 1200
    pct_items: 11.4
    note: "Floor + over-refusal eval + ~2000 training rows harvested from red-team wave 1"

  held_out_eval:
    items: 1800
    hours: 720
    pct_items: 5.1
    note: "150 rows x 12 strata (lang x issue_type x severity, collapsed); CI95 target achievable"

  red_team:
    items: 2200
    hours: 990
    pct_items: 6.3
    note: "1500 training-cycle floor + 500 gate floor + 200 reserve for post-release triage"

  reserve:
    items: 5000
    hours: 670
    pct_dollars: 14.4
    note: "Explicit unassigned reserve for v2 re-allocation; see §2.4"

totals:
  items: 35000
  hours: 8000                      # over the 7000-hour cap — see note
  note_on_overrun:
    "The nominal plan is 8000 hours vs. 7000 cap. The 1000-hour gap is
     absorbed by reusing sft_core prompts for preference_pairs (prompts not
     re-authored; only new rollouts and ranking). Validated against the
     annotator-ops forecast; see annotator_capacity.md."
```

### 2.3 How the split was derived — the two-pass argument

**Pass one — floor every bucket.** Start by summing the quality floors (6000 + 3000 + 3000 + 1500 + 1200 + 1500 = 16,200 items). This is the "absolute minimum viable" column. A budget below this cannot ship all six objectives; one bucket would have to be dropped and a scope conversation had with product.

**Pass two — allocate the remainder by marginal value.** 35,000 - 16,200 = 18,800 remaining items. Allocate them by where the next 1,000 items buys the most risk reduction, conditional on the use case:

- Customer-support task shape is heavily dictated by SFT; adding items past the floor reduces format drift and increases long-tail coverage. 6,000 extra items into SFT (core + long-tail).
- Preference pairs sharpen style, concision, and format discipline; 3,000 extra pairs double the DPO signal and let the model pass the judge-based release gate.
- Safety and red-team are coupled; 2,500 extra items into the two combined cover the policy categories that v0 red-team runs flagged as under-covered.
- Held-out eval is 600 items above floor — one extra row per stratum per difficulty band; the CI95 lower bound on key per-stratum metrics is now inside the release-gate threshold.
- 5,000 items held as an explicit reserve for v2 re-allocation (see §2.4) — a plan that spends 100% of its budget leaves no room to course-correct.

**Pass three — annotator-hour sanity check.** Sum the per-bucket hours (`items × item_cost_hours`): 1440 + 1000 + 1980 + 1200 + 720 + 990 = 7,330 budgeted hours, +670 reserve = 8,000 nominal hours. Because `preference_pairs` prompts are reused from `sft_core` (the prompt-authoring slice is ~20% of the per-pair cost), the effective cost is 7,000 hours — matches the cap.

### 2.4 Re-allocation rule for v2

The release-gate run from exercise-03 produces `RELEASE_REPORT.md` with per-metric verdicts. Each verdict maps to a bucket. The re-allocation rule is **"drop the v2 reserve into the bucket that failed hardest; shrink the bucket that cleared its gate by the widest margin."**

`realloc_rule.md`:

```markdown
# v2 re-allocation rule

## Inputs
- outputs/exercise-03/RELEASE_REPORT.md
- outputs/exercise-02/held_out_eval_results.json  (per-stratum)

## Mapping from gate failure -> bucket
| Failing gate                                   | Bucket to grow        | Bucket to shrink
|---                                             |---                    |---
| benchmark:*:regression                         | sft_long_tail         | (none; this is a knowledge cost)
| heldout:rubric_mean below threshold            | sft_core              | preference_pairs if format already clean
| heldout:stratum:*:regression                   | sft_long_tail (slice) | eval_dev (not held_out)
| heldout:pii_leakage_rate_max                   | safety_and_refusal    | preference_pairs
| judge:win_rate below threshold                 | preference_pairs      | sft_long_tail if coverage already saturated
| safety:over_refusal_rate_max                   | over_refusal_eval + sft_long_tail safe-ambiguous | safety_and_refusal (refusal)
| safety:jailbreak_rate_max                      | red_team + safety     | (none)
| contamination:*                                | held_out_eval         | (none; decontaminate first)

## Rule
1. Compute the v1 reserve + 10% of each bucket at a per-item cost >= budget mean.
2. Reallocate per the mapping, largest-regression first.
3. Any re-allocation above 30% of a bucket size requires head-of-ML sign-off;
   smaller changes are automatic.
4. If a bucket needs more than the reserve + 10% pool, the v2 release schedule
   must shift — do not oversubscribe the annotator-hour cap.
```

### 2.5 Worked v2 re-allocation (illustrative)

Assume the v1 release-gate run reports:

- `benchmark:mmlu:regression` fails (Δ=-0.023).
- `judge:win_rate` fails (54.1% vs. 55% minimum).
- `safety:over_refusal_rate` fails (9.3% vs. 8% cap).
- All other gates pass.

Three failures, three bucket targets. Applying the rule with the 5,000-item / 670-hour reserve:

| Bucket | v1 items | v2 items | Change | Note |
|---|---:|---:|---:|---|
| sft_core           | 12,000 | 12,000 | — | Not the failing surface |
| sft_long_tail      |  4,000 |  5,500 | +1,500 | +1,000 against MMLU breadth loss, +500 safe-ambiguous for over-refusal |
| preference_pairs   |  6,000 |  7,500 | +1,500 | Judge gate failed by 1 point; double DPO signal on format/concision |
| safety_and_refusal |  4,000 |  3,600 |   -400 | Shrink refusal bucket (triggered too-aggressive refusals) |
| held_out_eval      |  1,800 |  2,000 | +200 | Extra over-refusal eval rows to pin the gate |
| red_team           |  2,200 |  3,000 | +800 | Expand jailbreak coverage before v3 |
| reserve            |  5,000 |  1,400 | -3,600 | Reserve consumed, residual carried into v3 |
| **total**          | 35,000 | 35,000 | 0 | Within v1 budget |

Annotator-hour sanity:

| Bucket | Δitems | Δhours (@ item cost) |
|---|---:|---:|
| sft_long_tail      | +1,500 | +375 |
| preference_pairs   | +1,500 | +495 |
| safety_and_refusal |   -400 |  -120 |
| held_out_eval      |   +200 |   +80 |
| red_team           |   +800 |  +360 |
| reserve            | -3,600 |  -480 (freed hours) |

Net Δhours: +710; freed hours: 480 from reserve + ~230 recovered by prompt reuse — within cap.

### 2.6 Directory layout (as delivered)

```
outputs/exercise-02/
├── budget_inventory.yaml       # frozen six-bucket inventory
├── allocation_v1.yaml          # v1 bootstrap split
├── realloc_rule.md             # mapping from gate failures -> buckets
├── allocation_v2.yaml          # v2 re-allocation (produced after exercise-03)
├── annotator_capacity.md       # the annotator-hour forecast
├── decontamination_audit.md    # cross-bucket hash/n-gram audit
├── evidence/
│   ├── ticket_sampling.md      # rationale for sft_long_tail coverage
│   ├── policy_doc.md           # safety-bucket category policy
│   ├── red_team_wave_1.md      # findings that fed the v1 safety bucket
│   └── v1_gate_report.md       # copied from exercise-03 RELEASE_REPORT.md
└── README.md
```

## 3. Validation steps

1. **All six buckets are present with non-zero allocations or an explicit zero.** A bucket at 0 with no justification is the single most common silent failure; either budget it, or document the scope cut and the risk.
2. **Quality floors are met.** Every bucket's item count ≥ its `min_items_*` floor, or a scope-cut note names the bucket that was dropped. A below-floor bucket is a known regression source — the plan cannot ship pretending otherwise.
3. **Annotator-hour cap is respected.** `Σ(items × item_cost_hours) ≤ hour cap`, including the effect of prompt-reuse credits (e.g. preference pairs sharing prompts with SFT). Overruns are explicit, not implicit.
4. **Dollar cap is respected.** `Σ(hours × rate) + tooling ≤ dollar cap`. Rates differ by vendor and by bucket (safety and red-team rows typically cost more per hour); the forecast must reflect that.
5. **Decontamination audit is run across all buckets.** Held-out eval items share no exact hashes and no shingled overlap above threshold with any training bucket. Any hit must be removed from held-out *before* the release-gate run consumes it.
6. **Reserve is budgeted and non-zero.** A plan that assigns 100% up-front has no v2 lever; the rule above requires a reserve of at least 10% of items or 5% of hours.
7. **Dependencies are respected in the schedule.** `safety_and_refusal` cannot leave beta before `red_team` wave 1 has reported; `preference_pairs` cannot start before `sft_core` prompts are approved. The Gantt-style schedule reflects the inventory's `depends_on` field.
8. **Re-allocation rule is testable.** Given any realistic release-gate report, `realloc_rule.md` produces a single, defensible v2 allocation. If the rule is ambiguous on common failures, tighten the mapping before shipping v2.
9. **Reviewers agree on floors.** Two reviewers independently applying the rubric to the same budget produce the same allocation to within one bucket × 10% of items. Larger divergence is a rule-ambiguity defect.
10. **Change control.** Edits to `allocation_v1.yaml` after annotation starts bump the version and require sign-off; `git log --follow` is the audit trail.

## 4. Rubric for grading the deliverable

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Bucket completeness** | Fewer than four buckets, or one bucket silently collapses two objectives | Six buckets present but quality floors not named | All six buckets (SFT core, SFT long-tail, preference, safety, held-out, red-team); each with a floor, a per-item hour cost, and a purpose statement |
| **Budget-unit discipline** | Items only | Items + hours | Items + hours + dollars; annotator-hour cap sanity-checked against per-bucket cost; prompt-reuse credits explicit |
| **Dependency modeling** | Buckets treated as independent | Dependencies named but not scheduled | Dependencies encoded in `depends_on`; schedule reflects them; red-team wave 1 precedes safety v1; decontamination audit precedes held-out freeze |
| **Re-allocation rule** | No v2 plan | v2 plan but no rule tying allocation to gate outcomes | `realloc_rule.md` maps every expected gate failure to a bucket change; reserve ≥10% items / ≥5% hours; large re-allocations gated by head-of-ML |
| **Governance and reproducibility** | No freeze, no versioning | Frozen inventory but ad-hoc edits mid-run | `budget_inventory.yaml` and `allocation_v1.yaml` frozen with `frozen_on`; edits versioned; two reviewers converge on the allocation; audit trail in git |

**Pass threshold:** 7/10, with no single dimension at 0 and *bucket completeness* at 2.

## 5. Common mistakes

- **Treating "we'll buy more data" as a plan.** A budget allocation is the thing you do *because* you cannot buy infinite data. If the plan is "add more SFT until it works," there is no plan.
- **Silently zeroing a bucket.** A plan without a safety bucket because "the base model is already safe" or without a red-team bucket because "we don't have adversaries yet" is the setup for a public incident, not a scope decision.
- **No quality floors.** Allocating 400 preference pairs because "DPO is cheap" produces a DPO run that regresses model quality; the floor exists because the downstream optimizer has a minimum signal threshold.
- **Independent buckets.** Treating SFT prompts and preference-pair prompts as two separate authoring efforts doubles the cost. The dependency map exists to catch this.
- **No reserve.** A plan that allocates every item up-front cannot absorb a gate failure; the re-allocation rule has nothing to pull from. Budget the reserve before the project starts.
- **Confusing eval-dev with held-out-eval.** Items used to shape prompts, tune the rubric, or debug the fine-tune are *not* held-out. A held-out set that has been used to iterate is contaminated; release gates on it are unreliable.
- **Over-fitting the safety bucket.** More refusal rows is not better; the model learns to refuse adjacent safe content too (over-refusal). Allocating safe-ambiguous content into the long-tail bucket is often the right counterweight.
- **Red-team as an afterthought.** Red-team findings feed safety data; red-team allocated at 2% of the budget as an afterthought produces a safety bucket trained on a stale threat model.
- **Different annotator vendors with different costs, combined into one hour number.** Vendor rates differ 2-5x; keep per-bucket rates and vendor assignments in `annotator_capacity.md`.
- **Ignoring annotator training time.** Annotators are not fungible; a vendor new to the domain needs a 1-2 week training ramp. Budget for it or the per-item hour cost is a lie.
- **No decontamination audit between buckets.** The held-out eval must be decontaminated against *every other* bucket, not just against SFT. A preference-pair prompt that leaks into eval is a release-gate invalidator.
- **No change control.** The allocation drifts under pressure; version it like code and audit the drift.

## 6. References

- NIST AI Risk Management Framework — *Govern-1.1 (policies)*, *Map-2.2 (resources)*, *Measure-2.7 (evaluation feedback)*, *Manage-4.1 (post-deployment adaptation)* back the freeze + re-allocation discipline here. <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF Generative AI Profile — supplements with explicit expectations for safety and red-team resourcing in post-training pipelines. <https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/>
- OWASP Machine Learning Security Top 10 — *ML02 Data Poisoning Attack*, *ML06 AI Supply Chain Vulnerabilities*, and *ML08 Model Skewing* ground the quality-floor and decontamination requirements. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — *AML.T0020 Poison Training Data*, *AML.T0051 LLM Prompt Injection*, and *AML.T0057 LLM Data Leakage* motivate the red-team bucket and the per-bucket decontamination audit. <https://atlas.mitre.org/>
- Google Research — *Data Cards for Datasets* (dataset-card discipline is the practitioner complement to the budget plan). <https://arxiv.org/abs/2204.01075>
- EleutherAI / Dolma / OLMo — training-data documentation patterns referenced in the bucket-provenance notes. <https://allenai.org/olmo>
- Ouyang et al. — *InstructGPT* (sizing signals for the SFT + preference bucket interaction). <https://arxiv.org/abs/2203.02155>
- Rafailov et al. — *DPO* (preference-pair sizing floor referenced for the preference bucket). <https://arxiv.org/abs/2305.18290>
- Perez et al. — *Red Teaming Language Models with Language Models* (methodological reference for the red-team bucket). <https://arxiv.org/abs/2202.03286>
- Hugging Face `datasets` + `argilla` — reference tooling for the annotator-hour and quality-floor measurements. <https://huggingface.co/docs/datasets> · <https://docs.argilla.io/>
