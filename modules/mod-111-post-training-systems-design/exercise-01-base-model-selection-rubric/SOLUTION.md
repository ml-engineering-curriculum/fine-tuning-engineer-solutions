# mod-111 · exercise-01 — Base Model Selection Rubric — Solution

## 1. Solution overview

The deliverable is a **weighted, pre-registered rubric** that a team uses to pick a base model to fine-tune, plus a worked scoring of three candidate base-model families against one concrete use case. The rubric is the artifact; the scores and the decision are the demonstration that the rubric is usable.

Four ideas carry the exercise:

- **Base-model selection is an eight-axis decision, not a benchmark race.** Teams that rank candidates by MMLU alone ship models that fail on licensing, serving cost, or safety posture three months later. The rubric must force explicit scoring across *all* axes before an aggregate emerges.
- **Weights come from the use case, not from the model.** The same base-model shortlist can produce different winners for "customer-support chatbot" vs. "on-device code assistant." Freeze the weights before you score; otherwise the rubric degenerates into motivated reasoning.
- **"Fine-tune-ability" is a first-class axis.** Open weights, documented tokenizer, published chat template, a usable `config.json` for `transformers`/`vllm`, and a healthy ecosystem of PEFT recipes decide whether the team ships in six weeks or six months.
- **The rubric is a governance artifact, not a shopping list.** NIST AI RMF `Govern-1.1`, `Govern-1.4`, and `Map-1.1` all expect the organization to document the context, intended use, and risk assessment that led to a model choice — the rubric is exactly that documentation.

The worked example picks a base for the use case "English-only customer-support assistant for a mid-size SaaS, 4k-token context, deployed on 1× A100 per replica, must be offline-capable for air-gapped enterprise customers." Three families are scored: a mid-weight open-weights model family (Llama 3.x 8B), a mid-weight permissively-licensed European open-weights family (Mistral 7B/Nemo class), and an API-only proprietary family (reasoning via the managed provider).

## 2. Implementation — worked rubric and scoring

### 2.1 The rubric — eight axes, pre-registered weights

`base_model_rubric.yaml`:

```yaml
# base_model_rubric.yaml — version 1.0
# Freeze this file before scoring any candidate. Edits after scoring must bump
# the version and re-score all candidates on the new version.
version: "1.0"
frozen_on: "2026-11-24"
author: "ml-platform-lead@example.com"

use_case:
  name: "customer-support-assistant-v1"
  deployment: "self-hosted, 1xA100/replica, air-gap-capable"
  languages: ["en"]
  context_tokens_p95: 4096
  latency_budget_ms_p95: 1500
  data_residency: "customer-tenant; some EU; some on-prem"

# Axes are scored 1-5. An axis at 1 is grounds to reject the candidate even if
# the weighted total is high — see `knockouts` below.
axes:

  quality_on_task:
    weight: 0.20
    scoring_guide:
      1: "Base model below smallest open-weights peers on task-shaped evals"
      3: "Base model competitive with peers in its parameter class"
      5: "Base model clearly ahead of peers in its parameter class on the
          task-shaped eval set (held-out, decontaminated, CI-tested)"
    evidence_required:
      - "lm-eval-harness run against the task eval set from mod-108 ex-02"
      - "Point estimate, CI95, and the z-test vs. the next-best candidate"

  license_and_terms:
    weight: 0.15
    scoring_guide:
      1: "License prohibits the intended commercial use OR acceptable-use
          policy conflicts with the deployed product"
      3: "Permissive license (Apache-2.0 / MIT / BSD-3) OR community license
          with a clean carve-out for the intended use"
      5: "Permissive license with no AUP carve-outs that touch the product,
          no attribution requirement that leaks into runtime output, redistribution
          and derivative-weights rights unambiguous"
    evidence_required:
      - "Link to the license text and any AUP"
      - "Legal sign-off recorded in the rubric file"

  fine_tune_ability:
    weight: 0.15
    scoring_guide:
      1: "Weights not available OR only an API-side fine-tune product with
          no exportable artifact; tokenizer and chat template not published"
      3: "Open weights; documented tokenizer; community PEFT recipes exist
          but are not first-party; no official safety-tuning example"
      5: "Open weights; official HF repo with tokenizer, chat template, and
          generation_config; first-party SFT/PEFT/DPO recipe; healthy
          downstream adapter ecosystem on HF"
    evidence_required:
      - "Link to the HF repo tag that will be pinned"
      - "List of published recipes/adapters used as starting points"

  serving_cost:
    weight: 0.15
    scoring_guide:
      1: "Does not fit on target hardware at acceptable concurrency without
          multi-GPU sharding, or managed-API cost model is above budget"
      3: "Fits on target hardware with room for kv-cache at p95 context;
          acceptable concurrency with quantization"
      5: "Fits comfortably on target hardware at target concurrency with
          headroom for a 2x traffic spike; quantization path validated"
    evidence_required:
      - "Memory estimate at bf16 and 4-bit (mod-109 numbers)"
      - "Projected $/1k tokens or $/replica-hour at target load"

  safety_posture:
    weight: 0.10
    scoring_guide:
      1: "No published model card, no safety eval, known problematic refusal
          behavior or jailbreak reports, no provenance for safety tuning"
      3: "Published model card; vendor-run harmlessness evaluation; some
          jailbreak-resistance evidence; refusal behavior documented"
      5: "Published model card, independent safety evaluation (red-team
          reports, adversarial robustness, over-refusal metrics); documented
          safety-tuning recipe that can be reproduced or extended"
    evidence_required:
      - "Link to model card and any safety report"
      - "Internal red-team pass score, if a scoping run was done"

  ecosystem_and_tooling:
    weight: 0.10
    scoring_guide:
      1: "Minimal transformers/vllm/TGI support; no first-class quantization
          recipes; broken downstream adapters"
      3: "Supported by transformers and one production serving stack;
          GPTQ/AWQ/NF4 recipes exist; some adapters work"
      5: "First-class support in transformers, vllm, TGI; all three major
          quant stacks have validated recipes; wide adapter ecosystem"
    evidence_required:
      - "Smoke test: load, run a single prompt, serve under vllm"
      - "List of transforms (quant, LoRA merge, GGUF) that worked on this base"

  data_provenance:
    weight: 0.08
    scoring_guide:
      1: "No training-data disclosure; vendor refuses to answer provenance
          questions; evidence of copyright-infringing data"
      3: "Partial disclosure; filter policies described but not auditable;
          named datasets with licenses that match the intended use"
      5: "Detailed training-data disclosure including filter policies,
          source lists, dedup methodology, and a legal-hold statement for
          the training corpus"
    evidence_required:
      - "Training-data card or equivalent disclosure"
      - "Any public legal challenge against the vendor or training corpus"

  supply_chain_and_ops:
    weight: 0.07
    scoring_guide:
      1: "No published hash/signature; no model-card version history; no
          security-contact for weight-level issues"
      3: "Weights hosted on a vendor-controlled registry with a documented
          signing scheme; vulnerability disclosure channel exists"
      5: "Signed weights (sigstore / in-toto / SLSA-style); SBOM for
          tokenizer and runtime; vendor publishes security advisories;
          rollback path for revoked checkpoints"
    evidence_required:
      - "sha256 of the pinned weights checkpoint"
      - "Vendor security contact or VDP link"

# Knockouts — a candidate at 1 on any of these is rejected regardless of total.
knockouts:
  - license_and_terms         # a bad license cannot be out-scored by MMLU
  - data_provenance           # for regulated or enterprise deployments
  - fine_tune_ability         # if the plan is to fine-tune, closed weights is a knockout
  - serving_cost              # the model has to fit the deployment envelope

# Aggregate rule: weighted sum, then apply knockouts. An aggregate without a
# knockout check is not the decision; it is the sorting hint.
decision_rule:
  method: "weighted_sum_with_knockouts"
  tie_break_threshold: 0.3    # within 0.3 points, decide on qualitative notes
```

Two rules make this rubric more than a spreadsheet:

- **Freeze before scoring.** The same freeze discipline as the release-gates YAML. A rubric edited mid-scoring is not a rubric; it is motivated reasoning with extra steps. NIST AI RMF `Govern-1.1` requires documented policies for model selection; the freeze is the documentation hook.
- **Knockouts dominate the weighted sum.** A candidate that wins on quality but has a license that prohibits commercial redistribution is not a candidate. The eight axes are a *defense-in-depth* filter, not a beauty contest.

### 2.2 Scoring worksheet — three candidates against the frozen rubric

`scores.yaml`:

```yaml
# Scored on 2026-11-24 using base_model_rubric.yaml version 1.0.
# Each score links to evidence; no score is written without an evidence pointer.

candidates:
  - id: "llama-3.1-8b-instruct"
    source: "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
    scores:
      quality_on_task:       {value: 4, note: "task-shaped eval p3 of 3 peers; see ex01.md"}
      license_and_terms:     {value: 3, note: "Llama Community License; MAU<700M carve-out OK for this product; AUP reviewed by legal"}
      fine_tune_ability:     {value: 5, note: "published chat template, HF-official repo, DPO + SFT recipes, LoRA target modules documented"}
      serving_cost:          {value: 5, note: "fits on 1xA100 bf16 with 4k kv cache; 4-bit AWQ validated under vllm"}
      safety_posture:        {value: 4, note: "model card + Llama 3 safety report + independent red-team write-ups"}
      ecosystem_and_tooling: {value: 5, note: "first-class in transformers/vllm/TGI; GPTQ/AWQ/NF4/GGUF all validated"}
      data_provenance:       {value: 3, note: "Llama 3 training-data card; partial disclosure; no public infringement injunction as of freeze date"}
      supply_chain_and_ops:  {value: 4, note: "HF-hosted, revision pinned, sha256 recorded; vendor security contact exists"}
    weighted_total: 4.14
    knockouts_hit: []
    verdict: "shortlist"

  - id: "mistral-nemo-12b-instruct"
    source: "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407"
    scores:
      quality_on_task:       {value: 4, note: "edges 8B peers on task eval; slightly loses to larger 8B+ reasoning-tuned peers"}
      license_and_terms:     {value: 5, note: "Apache-2.0 for this release line; redistribution clean; no AUP carve-outs"}
      fine_tune_ability:     {value: 4, note: "published chat template; first-party SFT recipe; DPO recipe community-only"}
      serving_cost:          {value: 4, note: "12B params; bf16 fits on A100-80GB; 4-bit required for concurrency target"}
      safety_posture:        {value: 3, note: "model card published; limited independent red-team coverage"}
      ecosystem_and_tooling: {value: 4, note: "transformers/vllm first-class; GGUF path validated; GPTQ recipe present"}
      data_provenance:       {value: 3, note: "partial disclosure; vendor publishes dedup/filter approach"}
      supply_chain_and_ops:  {value: 4, note: "HF-hosted; revision pinnable; vendor publishes advisories"}
    weighted_total: 3.96
    knockouts_hit: []
    verdict: "shortlist"

  - id: "api-only-proprietary"
    source: "<managed vendor; API-only fine-tuning; no exportable weights>"
    scores:
      quality_on_task:       {value: 5, note: "state-of-class on the task eval"}
      license_and_terms:     {value: 2, note: "ToS forbids weight export; redistribution impossible"}
      fine_tune_ability:     {value: 1, note: "no exportable fine-tuned artifact; managed-API fine-tune only"}
      serving_cost:          {value: 2, note: "per-token pricing above budget at target throughput; no on-prem path"}
      safety_posture:        {value: 4, note: "mature safety pipeline"}
      ecosystem_and_tooling: {value: 3, note: "SDK excellent; cannot load in vllm/TGI"}
      data_provenance:       {value: 3, note: "partial disclosure under vendor NDA"}
      supply_chain_and_ops:  {value: 3, note: "vendor-controlled; model snapshots dated; revocation handled by vendor"}
    weighted_total: 2.98
    knockouts_hit: ["fine_tune_ability", "serving_cost"]
    verdict: "reject (two knockouts: cannot self-host; cannot export fine-tuned weights for air-gapped customers)"
```

### 2.3 Decision document — `DECISION.md`

```markdown
# Base-model selection — customer-support-assistant-v1

## Decision
Select **llama-3.1-8b-instruct** (HF revision pinned in `env.txt`) as the
base model for the customer-support-assistant v1 fine-tune.
Runner-up: **mistral-nemo-12b-instruct**.
Rejected: **api-only-proprietary** (two knockouts).

## Weighted totals (same rubric, same weights, same frozen version)
| Candidate | Total | Knockouts |
|---|---:|---|
| llama-3.1-8b-instruct      | 4.14 | — |
| mistral-nemo-12b-instruct  | 3.96 | — |
| api-only-proprietary       | 2.98 | license_and_terms (2), fine_tune_ability (1), serving_cost (2) |

## Rationale
- **Deployment envelope decides.** The air-gap requirement knocks out any
  API-only vendor; both open-weights candidates clear this.
- **Quality delta vs. cost delta.** Llama 3.1 8B and Mistral-Nemo 12B score
  within 0.2 on the weighted total; the two differ qualitatively in the
  license axis (Mistral is Apache-2.0, Llama is community-license with the
  MAU carve-out). Legal signed off on the Llama terms for this product, so
  the quality/fine-tune-ecosystem lead carries the decision.
- **Revisit trigger.** If the MAU carve-out ever becomes constraining, or
  if the Llama 3 line is ever revoked / superseded by a license change, the
  decision reverts to the runner-up without re-running the rubric. A
  license-change *is* a re-score event on axis 2 only; downgrade and apply
  the knockout check.

## Risks and follow-ups
1. **Carve-out drift.** The Llama community-license MAU threshold is a
   compliance watch item; product tracks MAU and legal re-reads the license
   quarterly.
2. **Quality regression at the next release.** If the Llama line ships a 3.x
   successor that regresses on the task eval, re-score quality and compare
   before adopting.
3. **Safety-posture drift.** If independent red-team reports surface new
   jailbreak classes on the chosen model, run the mod-106 ex-04 harness
   and re-score the safety axis.
4. **Managed-API threshold.** If a managed vendor ships an air-gap deployment
   option, the two knockouts above may lift; re-score before switching.
```

### 2.4 Directory layout (as delivered)

```
outputs/exercise-01/
├── base_model_rubric.yaml   # frozen rubric, v1.0
├── scores.yaml              # per-candidate scores with evidence pointers
├── evidence/
│   ├── task_eval/           # lm-eval JSON for each candidate on the task set
│   ├── serving/             # mod-109 memory + throughput numbers per candidate
│   ├── licenses/            # copies of each license text + legal sign-off
│   ├── safety/              # vendor model cards + any red-team reports
│   └── env.txt              # pins and HF revisions for every candidate
├── DECISION.md              # the signed decision
└── README.md
```

## 3. Validation steps

1. **The rubric is frozen before any candidate is scored.** `git log --follow base_model_rubric.yaml` shows the commit timestamp predates the earliest `scores.yaml` entry. A rubric edited after scoring is not the rubric — bump the version and re-score all candidates.
2. **Weights sum to 1.0.** A spot-check line in the rubric or a CI step asserts `sum(axis.weight for axis in axes) == 1.0`. Weights that do not sum to one produce weighted-total numbers that do not compare across candidates.
3. **Every score has an evidence pointer.** `scores.yaml` entries without a `note` or a file path in `evidence/` are rejected. "I think it's a 4" is not a score.
4. **Knockouts are applied after the weighted sum, not before.** A knockout rejects a candidate for the use case; it does not change the weighted-total number. The two numbers (total and knockouts-hit) travel together.
5. **The same evidence base is used for every candidate.** The task eval in `evidence/task_eval/` is the same eval set, same seed, same chat-template application across all candidates. Mixing eval versions across rows is the single most common invalidator of this rubric.
6. **Legal has signed off on `license_and_terms`.** A `license_and_terms: 5` score for a community-license model is a factual error; `5` requires permissive license with no AUP carve-out. Legal review is recorded in the note.
7. **Hardware envelope assumptions are explicit.** The `serving_cost` scores reference numbers from the mod-109 exercises (memory, throughput, quantization path). A score without a target-hardware anchor cannot be defended.
8. **The decision document names a runner-up and a revisit trigger.** `DECISION.md` without a runner-up is an incomplete deliverable — a base-model choice that cannot roll back is a single point of failure.
9. **The rubric survives weight sensitivity.** Re-run the weighted sum with ±20% on each axis weight. If the winner changes within this band, the decision is coin-flip; use the tie-break threshold and note it explicitly.
10. **Reproducible across reviewers.** Two reviewers scoring independently on the same evidence agree to within 1 point on every axis. Larger disagreements go back to the rubric for clarification, not to a tie-breaker vote.

## 4. Rubric for grading the deliverable

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Rubric completeness** | Fewer than six axes; weights not pre-registered | Eight axes present but weights incomplete or not frozen | All eight axes present (quality, license, fine-tune-ability, serving cost, safety posture, ecosystem, data provenance, supply chain); weights sum to 1.0; `frozen_on` recorded |
| **Knockouts and defense-in-depth** | Rubric is a weighted sum with no knockouts | Knockouts listed but applied inconsistently | Knockouts explicit for license, provenance, fine-tune-ability, serving cost; applied uniformly to every candidate; aggregate + knockouts travel together |
| **Evidence linkage** | Scores without pointers | Some scores evidenced; one or two "I think it's a 4" entries | Every score has an evidence pointer in `evidence/`; legal sign-off on license axis; task eval under `evidence/task_eval/` is the same set for every candidate |
| **Decision document** | Missing or score-only | Decision stated but no runner-up and no revisit trigger | Decision + rationale + runner-up + explicit revisit trigger (license change, line revoked, quality regression) + weight-sensitivity check |
| **Reproducibility and fairness** | Candidates scored against different eval sets or different hardware assumptions | Shared evidence but version pins drift between candidates | HF revisions pinned per candidate; same eval, same hardware, same chat-template application; two reviewers agree to ±1 point per axis |

**Pass threshold:** 7/10, with no single dimension at 0 and *rubric completeness* at 2.

## 5. Common mistakes

- **Scoring quality first, then back-filling the other axes.** The weighted sum you want to see on the winner determines the scores on the other seven axes. Freeze the rubric, then score each axis before looking at the aggregate.
- **No knockouts.** A rubric that only sums weighted scores cannot say "no, not for this deployment." Air-gap deployments reject API-only vendors; the rubric must encode that structurally.
- **Benchmark cherry-picking.** Vendor-reported MMLU scores on a base model do not predict performance on the deployed task distribution. The quality axis evidence is the task-shaped eval from mod-108 ex-02, not the vendor's blog post.
- **Ignoring license and AUP.** Permissive weights can carry an AUP that forbids the product's intended use; "open weights" does not equal "usable for this deployment." Legal sign-off is not optional.
- **Scoring "ecosystem" without a smoke test.** A model that advertises transformers/vllm support but fails to load under the pinned versions is a production hazard. Smoke-test every candidate on the deployment stack before scoring axis 6.
- **Weighting quality at 50%.** A rubric that leans this hard on one axis is a benchmark leaderboard in disguise. The maximum weight on any axis in a mature rubric is in the 15-25% range; the point is defense-in-depth.
- **No `frozen_on` date.** Without it, you cannot defend the rubric in a postmortem. Treat the rubric like the release gates in exercise-03: frozen before use, versioned, edits audited.
- **Skipping the data-provenance and supply-chain axes.** These are the OWASP ML Top 10 and MITRE ATLAS surfaces at base-model-selection time. Treating them as 7–8 points out of 100 is the correct weight; treating them as zero is the mistake.
- **No runner-up.** If the winner is revoked, re-licensed, or deprecated, the team needs a documented second choice that is already evidence-backed. The decision document must name it.
- **Weight sensitivity not checked.** Winners that depend on 0.20 vs. 0.22 on the quality axis are not winners; they are coin flips. Run the weight-sensitivity check before shipping the decision.
- **Scoring behind a closed door.** Two reviewers scoring on the same evidence catches both evidence gaps and rubric ambiguity. A single-reviewer rubric is a single point of judgment failure.
- **Treating the rubric as a one-time artifact.** Base-model lines ship new revisions regularly; a 3.1 → 3.2 → 4.x stream can shift any axis. The rubric is a living artifact — re-score on vendor-line revisions, not on vibe.

## 6. References

- NIST AI Risk Management Framework — *Govern-1.1 (policies and procedures)*, *Govern-1.4 (risk management culture)*, and *Map-1.1 (context establishment)* frame model selection as a governed, documented decision. <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF Generative AI Profile — extends Map/Measure for generative models; the base-model-provenance and supply-chain risks called out here map directly to axes 7 and 8. <https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/>
- OWASP Machine Learning Security Top 10 — *ML06 AI Supply Chain Vulnerabilities* is the supply-chain axis; *ML05 Model Theft* and *ML02 Data Poisoning* motivate the provenance disclosure and signed-weights requirements. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — the *ML Supply Chain Compromise* and *Discover ML Artifacts* technique categories are the threat model behind the supply-chain and provenance axes. <https://atlas.mitre.org/>
- Hugging Face Model Card guidance — model cards are the primary evidence surface for the quality, license, safety, and provenance axes; absence of a model card is a rubric red flag by itself. <https://huggingface.co/docs/hub/model-cards>
- Google Research — *Model Cards for Model Reporting* (Mitchell et al., 2019), the methodological antecedent of the model-card axis. <https://arxiv.org/abs/1810.03993>
- SLSA framework — levels and attestation model that back the "signed weights" expectation in axis 8. <https://slsa.dev/>
- in-toto / Sigstore — reference implementations of signed-artifact provenance usable for model weights. <https://in-toto.io/> · <https://www.sigstore.dev/>
- EleutherAI `lm-evaluation-harness` — the harness used to produce the task-eval evidence in `evidence/task_eval/`. <https://github.com/EleutherAI/lm-evaluation-harness>
- Llama community license (reference for the license-and-AUP discussion in the worked example). <https://llama.meta.com/llama3/license/>
- Apache License 2.0 (reference for the Mistral entry in the worked example). <https://www.apache.org/licenses/LICENSE-2.0>
