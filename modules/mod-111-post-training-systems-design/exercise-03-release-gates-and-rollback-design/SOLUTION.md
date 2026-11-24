# mod-111 · exercise-03 — Release Gates and Rollback Design — Solution

## 1. Solution overview

The deliverable is a **gate + rollback design** for shipping a post-trained model to production: a staged rollout with pre-committed gates at each stage, a rollback mechanism that can be invoked in under five minutes, and a runbook that names who pushes which button under which signal. The design is the artifact; the worked example demonstrates it against the customer-support-assistant from exercises 01–02.

Four ideas carry the exercise:

- **Rollout is multi-stage, not a flag flip.** Shadow → canary → staged ramp → full. Each stage has entry and exit gates; each stage runs long enough for the slowest-moving signal (quality regression, safety drift) to surface.
- **Automatic rollback is a mechanism; manual rollback is a decision.** The kill switch exists so a human can pull it in one command; automatic rollback exists so no human needs to be awake when a red-line gate fires.
- **Gates compose, failures escalate.** Automatic rollback for red-line gates (safety, PII, hard regressions). Pager for warning gates. Weekly report for drift gates. Mixing these levels produces alert fatigue and real incidents get missed.
- **Reproducibility is the audit's receipt.** Every rollout state (candidate tag, baseline tag, routing weights, gate values) is observable, timestamped, and reconstructable from the artifact alone. NIST AI RMF `Manage-2.3`, `Manage-2.4`, and `Govern-1.6` make this an explicit requirement.

The worked example ships `llama-3.2-3b-sft-v2` behind the existing `llama-3.2-3b-sft-v1-prod` routing key, through four rollout stages, with an automatic rollback on three red-line gates and manual-pager rollbacks on five warning gates.

## 2. Implementation — worked design

### 2.1 Rollout stages

| Stage | Traffic | Duration (minimum) | Exit gates (must all pass) | Rollback trigger (any one) |
|---|---|---|---|---|
| 0. Shadow                  | 0% live; 100% mirror | 24h | shadow quality parity (judge win-rate ≥ baseline −2%); latency p95 within SLO; no safety red-line; no cost regression | shadow safety red-line; serving failure rate > 1% |
| 1. Canary (internal)       | 100% of internal tenants | 24h | internal CSAT no worse than baseline; no PII leak reports; no sev-1/2 incidents | PII report; sev-1 incident; p99 latency > 2x SLO |
| 2. Staged ramp             | 1% → 5% → 25% → 50% external | 24h per step | per-step: automatic metrics all green; thumbs-down rate within +0.5 percentage point of baseline; no safety red-line | automatic-rollback gates; two consecutive step aborts |
| 3. Full                    | 100% external | — | 7-day watch window with drift gates armed | drift breach beyond cap for 3 consecutive hours; manual pager |

Two rules:

- **Entry gates are pre-committed.** The thresholds live in `release_gates.yaml` (from mod-108 ex-05), frozen before the candidate is trained. A candidate that does not pass the entry gate for stage 0 does not enter stage 0.
- **Exit gates are a time-and-signal pair.** A step that passes its metric gates in the first hour still has to serve the minimum duration before advancement. "It looks good" is not a gate; "all signals held green for the minimum duration under representative load" is.

### 2.2 Gate specification — `release_gates.yaml`

```yaml
# release_gates.yaml — version 2.0 (extends mod-108 ex-05 gates with rollout gates)
version: "2.0"
frozen_on: "2026-11-24"
candidate_tag: "llama-3.2-3b-sft-v2"
baseline_tag:  "llama-3.2-3b-sft-v1-prod"

# ---- Offline gates (from mod-108 ex-05, enforced before shadow) ----
offline:
  standard_benchmarks: { ... }      # unchanged from mod-108 ex-05
  held_out_task:       { ... }
  judge:               { ... }
  contamination:       { ... }
  safety_redlines:     { ... }

# ---- Online gates (added for the rollout) ----
online:
  stage0_shadow:
    duration_min_hours: 24
    entry:
      requires_all_offline_gates_passed: true
    exit_all:
      shadow_judge_delta_min: -0.02      # judge win-rate loss cap: 2 points
      shadow_latency_p95_ms_max: 1500
      shadow_cost_per_1k_tokens_delta_max: 0.10
    rollback_any:
      shadow_safety_redline_hit: true
      shadow_error_rate_max: 0.01

  stage1_canary_internal:
    duration_min_hours: 24
    traffic_pct: 100                    # 100% of internal tenants
    exit_all:
      internal_thumbs_up_rate_delta_min: -0.01
      internal_p95_latency_ms_max: 1500
      internal_sev1_count_max: 0
      internal_pii_leakage_count_max: 0
    rollback_any:
      internal_sev1_count_min_rollback: 1
      internal_pii_leakage_count_min_rollback: 1
      internal_p99_latency_ms_max: 3000

  stage2_ramp:
    steps_pct: [1, 5, 25, 50]
    duration_min_hours_per_step: 24
    exit_all:
      per_step_thumbs_down_delta_max: 0.005
      per_step_judge_delta_min: -0.02
      per_step_cost_per_1k_tokens_delta_max: 0.10
      per_step_refusal_rate_delta_max: 0.03
    rollback_any:
      step_pii_leakage_count_min_rollback: 1
      step_safety_redline_hit: true
      step_p99_latency_ms_max: 3000
      step_automatic_rollback_from_offline_gate: true

  stage3_full:
    watch_window_days: 7
    drift_gates:
      daily_thumbs_down_rate_delta_max: 0.005
      daily_refusal_rate_delta_max: 0.03
      daily_cost_delta_pct_max: 10
      daily_tokens_per_request_delta_pct_max: 10
    rollback_any:
      drift_gate_breach_consecutive_hours: 3

# ---- Rollback modes ----
rollback:
  modes:
    automatic:
      triggers:
        - "stage0_shadow.rollback_any"
        - "stage1_canary_internal.rollback_any"
        - "stage2_ramp.rollback_any (red-line subset)"
      action: "flip router baseline_tag -> served; page oncall; freeze candidate_tag"
      max_time_to_revert_minutes: 5

    manual_pager:
      triggers:
        - "drift gate breach"
        - "weekly report anomalies"
        - "external customer report"
      action: "oncall decides between hold, reduce traffic, or revert; decision + rationale recorded in incident doc"
      max_time_to_decision_minutes: 30

approvals:
  stage_transitions_require:
    - role: eval_owner
    - role: product_owner
  rollback_execution_requires:
    automatic: []                        # no human approval needed
    manual_pager:
      - role: oncall
      - role: eval_owner                 # async acknowledgement within 24h
```

### 2.3 Rollback mechanism — router + feature flag + kill switch

The cheapest rollback that works is **a routing table in front of the inference stack**. The product addresses one logical model name (`customer-support-assistant`); the router maps it to a tagged weight set. Rollback is one atomic write.

`router_config.yaml`:

```yaml
# router_config.yaml — read by the inference gateway
model_routes:
  customer-support-assistant:
    primary_tag:     "llama-3.2-3b-sft-v2"
    fallback_tag:    "llama-3.2-3b-sft-v1-prod"
    routing_weights: { v2: 100, v1: 0 }    # % of traffic per tag; must sum to 100
    canary_tenants:  []                    # list of tenant ids pinned to primary
    kill_switch:     false                 # if true, all traffic goes to fallback_tag immediately
```

Three mechanics:

- **Atomic routing update.** `routing_weights` is a single config key; write-through to the gateway with a version. The old and new weights do not coexist mid-update — either the gateway serves the old table or the new one, never both.
- **Kill switch bypasses weights.** `kill_switch: true` forces 100% to `fallback_tag` regardless of `routing_weights`; this is the "one-command revert" path. A healthy rollout never touches it; a red-line incident always does.
- **Tenant pinning for canary.** `canary_tenants` is the mechanism for stage 1 (internal only) and for staged-ramp pin-lists; the gateway resolves by tenant id first, then by weights.

`revert.sh`:

```bash
#!/usr/bin/env bash
# revert.sh — one-command rollback; invoked by the automatic trigger or by a human.
set -euo pipefail

REASON="${1:?usage: revert.sh <reason-string>}"
TS="$(cat /proc/uptime | awk '{printf "%.0f\n", $1}')s-uptime-stamp"
INCIDENT_DIR="incidents/rollback-${TS}"
mkdir -p "${INCIDENT_DIR}"

# 1. Snapshot the current router config.
cp router_config.yaml "${INCIDENT_DIR}/router_config.before.yaml"

# 2. Build the reverted config: kill_switch=true, weights zeroed on primary.
python tools/revert_router.py \
    --in router_config.yaml \
    --out router_config.reverted.yaml

# 3. Atomic swap via the config store (any KV store that supports CAS works).
tools/config_push.py --path router_config.yaml \
    --value "@router_config.reverted.yaml" --cas

# 4. Record the incident and page the oncall + evaluations-lead.
cat > "${INCIDENT_DIR}/incident.md" <<EOF
# Rollback executed
Reason: ${REASON}
Baseline served: llama-3.2-3b-sft-v1-prod
Candidate frozen: llama-3.2-3b-sft-v2
Config before: router_config.before.yaml
Config after: router_config.reverted.yaml
EOF

tools/page.py --team oncall --incident "${INCIDENT_DIR}"
tools/page.py --team evaluations --incident "${INCIDENT_DIR}"
```

The design is deliberately boring: a config swap under CAS, an incident doc, two pages. No ML in the rollback path — the rollback is a gateway change.

### 2.4 Observability — the signals the gates read

Each gate above names a metric; the rollout is only honest if the signals are instrumented before stage 0 starts. Minimum surface:

- **Quality**: shadow judge win-rate (hourly); internal CSAT / thumbs-up rate (hourly); held-out eval (daily on a 500-prompt rolling slice — the full held-out is reserved for release-gate reruns).
- **Safety**: refusal-rate, over-refusal-rate, PII-leak-flag count (per 1k requests); jailbreak-pattern detection counts (via the mod-106 ex-04 harness running in shadow).
- **Serving**: p50/p95/p99 latency, throughput (req/s), error rate, token-per-request, cost-per-1k-tokens.
- **Product**: thumbs-down rate, retry rate, escalation-to-human rate, abandonment rate.

All metrics are tagged with `model_tag` so baseline vs. candidate can be compared per-tenant per-hour. Any metric that cannot be split by `model_tag` cannot be a gate.

### 2.5 Runbook — `ROLLOUT_RUNBOOK.md`

```markdown
# Rollout runbook — llama-3.2-3b-sft-v2

## Who does what

| Role          | Pre-stage action                                 | During stage           | On incident
|---            |---                                               |---                     |---
| eval_owner    | Confirm offline gates green; sign stage transition | Watches dashboards every 2h | Confirms rollback classification; files postmortem
| product_owner | Confirm product telemetry wired; sign stage transition | Watches thumbs/CSAT | Decides whether to pause vs. revert
| oncall        | On-call page coverage confirmed                  | Pages received         | Runs `revert.sh` on red-line pages; drafts incident doc
| platform      | Confirms router + feature-flag ready             | Watches error rate     | Rolls routing; validates served tag matches config
| head_of_ml    | —                                                | —                      | Required signoff to re-advance after a rollback

## Stage advancement protocol
1. Monitor the stage's dashboards for the `duration_min_hours` window.
2. At the end of the window, auto-generate `STAGE_REPORT.md` from the gate values.
3. Both `eval_owner` and `product_owner` sign off; the signoff is a commit on the release branch.
4. Platform bumps the `routing_weights` and the gateway acknowledges the new config.

## Rollback decision tree
- **Red-line gate fires?** `revert.sh` runs automatically; oncall and eval_owner paged. No approval needed.
- **Warning gate fires?** Oncall is paged. Decision is one of:
  (a) pause the ramp at the current step,
  (b) reduce traffic to the previous step,
  (c) execute `revert.sh` manually.
  Decision + rationale recorded in `incidents/<ts>/incident.md` within 30 minutes.
- **Drift gate breach after full-rollout?** Weekly review decides; a 3-consecutive-hour breach triggers an oncall page.

## Post-rollback actions
1. `revert.sh` ran — `router_config.reverted.yaml` committed.
2. `incident.md` filled within one hour; postmortem within one week.
3. Candidate is frozen in `registry`; a re-release requires head_of_ml signoff.
4. Gate values that triggered the rollback are not loosened until the underlying failure is fixed and re-validated.
```

### 2.6 Worked rollout — `STAGE_REPORT_shadow.md` (illustrative)

```markdown
# Stage 0 shadow — llama-3.2-3b-sft-v2 vs. llama-3.2-3b-sft-v1-prod
Duration: 2026-12-02T14:00Z → 2026-12-03T14:12Z (24h 12m)
Candidate traffic: 0% live (shadow only)

## Gates

| Gate                                     | Threshold          | Observed           | Verdict
|---                                       |---                 |---                 |---
| shadow_judge_delta_min                   | ≥ -0.02 (win-rate) | -0.008             | pass
| shadow_latency_p95_ms_max                | ≤ 1500 ms          | 1240 ms            | pass
| shadow_cost_per_1k_tokens_delta_max      | ≤ +0.10            | +0.04              | pass
| shadow_safety_redline_hit                | 0                  | 0                  | pass
| shadow_error_rate_max                    | ≤ 0.01             | 0.0023             | pass

Verdict: **ADVANCE to stage 1 (internal canary)**.

Signoffs:
- eval_owner  — evaluations-lead@example.com — commit a3d1c4b
- product_owner — product-lead@example.com — commit a3d1c4b
```

### 2.7 Directory layout (as delivered)

```
outputs/exercise-03/
├── release_gates.yaml                    # offline + online gates
├── router_config.yaml                    # current routing table
├── revert.sh                             # one-command rollback
├── ROLLOUT_RUNBOOK.md                    # who/what/when
├── tools/
│   ├── revert_router.py                  # builds the reverted config
│   ├── config_push.py                    # CAS write to config store
│   └── page.py                           # pager integration
├── dashboards/
│   ├── quality.json
│   ├── safety.json
│   └── serving.json
├── stage_reports/
│   ├── STAGE_REPORT_shadow.md
│   ├── STAGE_REPORT_canary.md
│   ├── STAGE_REPORT_ramp_01pct.md
│   └── STAGE_REPORT_ramp_05pct.md
├── incidents/                            # one subdirectory per rollback
└── README.md
```

## 3. Validation steps

1. **Gates are frozen before the candidate is trained.** `git log --follow release_gates.yaml` shows the YAML's most-recent-edit commit predates the training run; edits during a stage bump the version and require sign-off.
2. **Rollback runs in under five minutes end-to-end.** Time `revert.sh` against a staging gateway; from invocation to 100% baseline-tag served must be measured, not estimated. "Fast" is not a number.
3. **The kill switch does not depend on the ML stack.** Flipping `kill_switch: true` must succeed even if the candidate weights are corrupted, the inference workers are OOM-ing, or a model-server is in a crash-loop. The router is the thing you trust.
4. **Automatic rollback fires on red-line gates only.** The automatic trigger list is bounded; warning and drift gates page a human. Pulling every gate into automatic rollback produces thrash; pulling none into it produces 3 AM incidents with no humans.
5. **Every metric the gates read is split by `model_tag`.** Dashboards show baseline vs. candidate side-by-side per-tenant per-hour. A gate on an un-tagged aggregate is unfalsifiable.
6. **Shadow is 100% mirror, not a sample.** Mirrored traffic to the candidate for the full duration; sampled shadow underestimates tail risks (long-context prompts, rare issue types).
7. **Stage transitions require two signoffs.** `eval_owner` and `product_owner`; recorded as commits on the release branch. A single signoff is not a stage transition.
8. **Incidents produce artifacts.** Every rollback writes `router_config.before.yaml`, `router_config.reverted.yaml`, and `incident.md`. The artifacts are the audit trail.
9. **Gates cannot be loosened during a stage.** The `version` on the YAML is monotonic; a change between stages bumps the version and resets the clock on affected stages.
10. **Rollback is routinely rehearsed.** Monthly fire-drill: trigger a shadow-stage rollback against a clone of production. Measure `revert.sh` wall-clock and page-receipt times; file any slippage as a sev-3 operational bug.

## 4. Rubric for grading the deliverable

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Staging + gates** | Single flag-flip rollout; no gates or thresholds | Multi-stage design but gates not pre-committed or missing per-stage duration | Four stages (shadow / canary / ramp / full); per-stage entry + exit + rollback gates; durations enforced; YAML frozen |
| **Rollback mechanism** | No kill switch; rollback requires rebuild or redeploy | Kill switch exists but takes > 15 minutes or depends on the ML stack | Router + kill switch; atomic swap under CAS; < 5 min end-to-end; mechanism independent of the ML stack |
| **Automatic vs. manual escalation** | Everything manual OR everything automatic | Classification exists but overlapping triggers | Red-line gates → automatic; warning gates → page; drift gates → weekly; non-overlapping triggers |
| **Observability** | No per-model-tag splits; gates on unobservable metrics | Metrics exist but not all gates are backed by dashboards | Every gate traces to a dashboard; every metric split by `model_tag`; shadow is 100% mirror; dashboards survive a candidate rollback |
| **Governance + reproducibility** | No signoffs; no incident artifacts | Signoffs defined but incidents not audited | Two signoffs per stage; incident artifacts (`before`, `reverted`, `incident.md`); monthly rollback fire-drill; gate edits versioned with `frozen_on` |

**Pass threshold:** 7/10, with no single dimension at 0 and *rollback mechanism* at 2.

## 5. Common mistakes

- **Rollback by redeploy.** A rollback that waits for a model server to re-pull weights is minutes, sometimes tens of minutes; the design is a router swap, not a rebuild.
- **Automatic rollback on every gate.** Flaky signals cause thrash; the model lands back on baseline, a human un-rollbacks to retest, repeat. Automatic rollback is reserved for red-line gates whose false-positive rate is near zero (PII leak, safety redline, critical regression).
- **Shadow by sampling.** A 5% shadow misses the rare prompts that will wake someone up later. If the mirror is expensive, scope it to representative long-context and high-risk categories, not a random sample.
- **Gates on un-split metrics.** "Latency is up" tells you nothing if you cannot split baseline vs. candidate. Every gate metric must carry `model_tag`.
- **No tenant pinning.** Enterprise customers often require either "always on baseline" or "always on candidate" for regulated workloads. Without pinning, the ramp cannot honor contracts.
- **Kill switch depends on model health.** If the kill switch is a model-server command instead of a router config, a crashlooping model-server defeats the switch. Push the switch out one layer.
- **Incident artifacts live in chat.** A rollback whose rationale lives in a Slack message cannot be audited six months later. The incident doc is the receipt.
- **No fire-drills.** A rollback mechanism that has never been tested will fail in production; test it monthly on a clone or on an idle region.
- **Mixing warning and red-line pagers.** If a drift breach wakes the same person as a PII leak, the on-call treats everything as noise or gets pulled into the wrong bucket.
- **Loosening a gate mid-stage.** The YAML is versioned; loosening mid-stage is a decision taken outside the pipeline and recorded as a signed override, not an edit.
- **Ignoring cost gates.** A model that is 30% better on quality but doubles token-per-request may be unshippable at scale. `cost_per_1k_tokens_delta_max` is a legitimate red-line for throughput-sensitive products.
- **No drift gates post-rollout.** "Shipped" is not "safe forever." Drift gates armed for at least 7 days post-rollout catch the slow regressions that per-step gates could not see.

## 6. References

- NIST AI Risk Management Framework — *Manage-2.3 (incident response)*, *Manage-2.4 (change management)*, *Govern-1.6 (accountability)*, and *Measure-3.1 (performance monitoring)* back the staged-rollout + rollback + incident artifact posture. <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF Generative AI Profile — the generative AI subcategories add explicit expectations for red-line safety rollbacks and post-deployment monitoring. <https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/>
- OWASP Machine Learning Security Top 10 — *ML05 Model Theft*, *ML06 AI Supply Chain Vulnerabilities*, and *ML09 Output Integrity Attack* motivate the kill-switch-independent-of-ML-stack and the signed-rollback-artifact requirements. <https://owasp.org/www-project-machine-learning-security-top-10/>
- OWASP LLM Top 10 for Large Language Model Applications — *LLM01 Prompt Injection* and *LLM06 Sensitive Information Disclosure* back the PII-leak and jailbreak-pattern red-line gates. <https://owasp.org/www-project-top-10-for-large-language-model-applications/>
- MITRE ATLAS — *Discover ML Artifacts*, *ML Supply Chain Compromise*, and *LLM Prompt Injection* shape the rollback runbook's threat model and the incident-artifact capture. <https://atlas.mitre.org/>
- Google SRE book — chapters on *Canarying*, *Progressive Rollouts*, and *Postmortems* ground the staged-rollout vocabulary and the signoff + incident-doc pattern. <https://sre.google/sre-book/table-of-contents/>
- Google — *Responsible AI Practices: Transparency and Accountability* (dual-signoff releases). <https://ai.google/responsibility/responsible-ai-practices/>
- Microsoft Responsible AI Standard — includes release-gate and rollback expectations for generative AI deployments. <https://www.microsoft.com/en-us/ai/responsible-ai>
- EleutherAI `lm-evaluation-harness` — the harness consumed by the offline gates. <https://github.com/EleutherAI/lm-evaluation-harness>
- Progressive delivery / feature-flag references — Flagger, Argo Rollouts, and OpenFeature for the underlying routing mechanic. <https://flagger.app/> · <https://argo-rollouts.readthedocs.io/> · <https://openfeature.dev/>
