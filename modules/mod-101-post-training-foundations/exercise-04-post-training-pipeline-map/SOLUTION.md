# mod-101 · exercise-04 — Post-Training Pipeline Map — Solution

## 1. Solution overview

A post-training pipeline map is the fine-tuning engineer's end-to-end schematic: the stages a base model moves through from "raw open-weights checkpoint" to "released adapted model with a review trail." The purpose of the map is operational: it is what the engineer points to in design review, in incident response, and in a NIST AI RMF-aligned [NIST-AI-RMF] governance narrative to say "this is where data was changed, this is where behavior was changed, this is the control that proved the change was safe."

A good map:

1. **Names each stage** and the artifact it produces.
2. **Names the decision** that moves work from one stage to the next (the gate).
3. **Names the risks** at each stage with reference to a published taxonomy (OWASP ML Top 10 [OWASP-ML] and MITRE ATLAS [ATLAS]).
4. **Names the governance function** each gate exercises (MAP, MEASURE, MANAGE, GOVERN under NIST AI RMF [NIST-AI-RMF]).

This solution gives a worked map, the decision rationale for the stage boundaries, and a rubric.

## 2. Implementation — worked map

### 2.1 The stages

```
  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
  │ 0 BASE MODEL  │───▶│ 1 DATA CURATE │───▶│ 2 SFT         │
  │  intake/audit │    │  +eval splits │    │  (supervised) │
  └───────────────┘    └───────────────┘    └───────────────┘
                                                    │
                                                    ▼
  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
  │ 5 EVAL GATE   │◀───│ 4 SAFETY TUNE │◀───│ 3 PREF OPT    │
  │ (held-out)    │    │ (red-team +   │    │ (DPO/ORPO/    │
  │               │    │  refusal mix) │    │  RLHF)        │
  └───────────────┘    └───────────────┘    └───────────────┘
          │
          ▼
  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
  │ 6 QUANTIZE +  │───▶│ 7 SERVE +     │───▶│ 8 MONITOR +   │
  │   EXPORT      │    │   RELEASE     │    │   ROLLBACK    │
  └───────────────┘    └───────────────┘    └───────────────┘
```

Every arrow is a **gate** — a checked decision, not a pipeline step. Gates fail closed.

### 2.2 Per-stage detail

| Stage | Inputs | Artifact produced | Gate to next stage | Primary risk (taxonomy) | NIST AI RMF function |
|---|---|---|---|---|---|
| 0 Base intake | Candidate open-weights checkpoint + its license + model card | `model-intake.md`: provenance, license grant, known limitations | License + provenance cleared; known-limitations acknowledged | Supply-chain compromise (ATLAS AML.T0010; OWASP ML06 AI Supply Chain Attacks) | MAP: context + risks framed |
| 1 Data curation + eval splits | Source corpora + licenses | Train / val / held-out eval sets with license trace + decontamination report | Eval set is truly held out; train set license-clean; de-duplicated vs. eval | Data poisoning (OWASP ML02); eval contamination | GOVERN (data policy), MEASURE (eval integrity) |
| 2 SFT | Stage-1 training set + base model | SFT checkpoint; training report; label-mask diagnostics | Loss on correct spans only; val loss curve sane; no catastrophic forgetting on general benchmarks | Overfit, forgetting, label leak | MANAGE |
| 3 Preference optimization | SFT checkpoint + preference pairs | DPO/ORPO/RLHF-tuned checkpoint; preference report | Preference signal is calibrated (chosen > rejected on held-out pairs); KL to SFT not blown out | Reward hacking; KL collapse | MANAGE |
| 4 Safety tuning | Stage-3 checkpoint + refusal-curated mix + red-team prompts | Safety-tuned checkpoint; red-team report | Harmful-output rate below release threshold; no over-refusal regression | Model evasion (ATLAS AML.TA0008); jailbreaks | MANAGE + MEASURE |
| 5 Eval gate | Stage-4 checkpoint + held-out eval suite | Release eval report | All release-blocking evals pass; delta vs. previous release reported | False confidence from narrow evals (OWASP ML04 Model Skewing) | MEASURE |
| 6 Quantize + export | Released-eligible checkpoint | Quantized artifact (GGUF, AWQ, GPTQ, int8, etc.) + parity report | Quantized vs. fp16 eval delta within budget | Quantization-induced drift | MEASURE |
| 7 Serve + release | Quantized artifact + runtime config | Served model version + model card update | Load test passes; model card published; rollback target identified | Model theft via query (ATLAS AML.T0024; OWASP ML05 Model Theft); PII leak | GOVERN (release), MANAGE |
| 8 Monitor + rollback | Live traffic | Drift report; incident tickets | Breach of threshold triggers rollback | Distribution drift; prompt injection in use (OWASP ML01 Input Manipulation) | MANAGE |

The four NIST AI RMF functions (GOVERN, MAP, MEASURE, MANAGE) are defined in the NIST AI Risk Management Framework 1.0 [NIST-AI-RMF]. OWASP ML Top 10 category labels and MITRE ATLAS technique ids above come from the published catalogs and are used here as lenses, not quantitative risk scores.

### 2.3 Why this shape — decision rationale

The sequence SFT → preference optimization → safety tuning → eval is not arbitrary. The learner should defend these specific orderings:

1. **Data curation is a stage, not a sub-task of SFT.** Treating curation as a notebook aside is where provenance and eval decontamination quietly rot. The curation artifact is independently reviewed and dated [OWASP-ML, ML02].
2. **SFT precedes preference optimization.** DPO/RLHF with no SFT prior tends to be unstable: the preference objective rewards relative ordering, which is poorly defined when the base has no task competence. Standard recipes (InstructGPT [IGP22], DPO [DPO23], Llama [L3]) run SFT first.
3. **Safety tuning is its own stage after preference optimization.** Folding refusal data into SFT dilutes the task signal; folding it into the preference stage interleaves two optimization targets with different KL constraints. Keeping it separate lets the fine-tuner explicitly measure *over-refusal regression* as a release gate.
4. **A held-out eval gate sits after all weight-changing stages.** Any eval set the model has seen weights-updated against cannot block its own release.
5. **Quantization is downstream of release decisions, not upstream.** Quantize an approved checkpoint; do not re-litigate safety on the quantized artifact except for parity (because the risk surface shifts).
6. **Monitoring + rollback is a stage, not an afterthought.** A release without a defined rollback target is a release without an undo button. The NIST AI RMF MANAGE function explicitly covers this [NIST-AI-RMF].

### 2.4 Artifacts a reviewer should see

The map is operational only if each stage produces something a reviewer can read:

- Stage 1 → `datasets/<mix>/CARD.md`: sources, licenses, decontamination method, counts.
- Stage 2 → `runs/<id>/training-report.md`: config, loss curves, label-mask sample, train/val split hash.
- Stage 3 → `runs/<id>/preference-report.md`: preference-pair source, held-out preference accuracy, KL to SFT.
- Stage 4 → `runs/<id>/safety-report.md`: red-team prompt set version, pre/post refusal rates, over-refusal control set.
- Stage 5 → `release/<version>/eval-report.md`: eval suite id, numbers vs. previous release, pass/fail per gate.
- Stage 6 → `release/<version>/quantization-parity.md`: fp16 vs. quantized deltas per eval.
- Stage 7 → `release/<version>/MODEL_CARD.md`: card diff vs. previous; rollback target named.
- Stage 8 → `monitor/<version>/drift-report.md`: live metrics, thresholds, incidents, rollback actions taken.

### 2.5 Risk-to-gate traceability

For each OWASP ML Top 10 category the student should be able to point to the gate(s) that catch it [OWASP-ML]:

| OWASP ML risk | Primary gate(s) | Supporting evidence |
|---|---|---|
| ML01 Input Manipulation | Gate 4 (safety), Gate 8 (monitor) | Red-team report + drift report |
| ML02 Data Poisoning | Gate 1 (data) | Data card + poisoning scan output |
| ML03 Model Inversion | Gate 7 (serve) | Rate limits + content filters |
| ML04 Membership Inference | Gate 1 (data), Gate 5 (eval) | Decontamination + eval set isolation |
| ML05 Model Theft | Gate 7 (serve) | Rate limits + access controls |
| ML06 AI Supply Chain Attacks | Gate 0 (intake) | Provenance + checksum record |
| ML07 Transfer Learning Attack | Gate 0 (intake), Gate 2 (SFT) | Base-model risk review; eval delta |
| ML08 Model Skewing | Gate 5 (eval) | Diverse held-out evals |
| ML09 Output Integrity Attack | Gate 7 (serve) | Signed responses / output validation |
| ML10 Model Poisoning | Gate 2–4 (training) | Training report + config review |

A single gate can cover multiple risks; the point of the table is that *every* OWASP ML Top 10 category has a named gate.

## 3. Validation steps

An acceptable submission shows:

1. **The nine stages (0–8) explicitly listed**, each with an input, an artifact, and a gate.
2. **A decision rationale for the ordering** — not just "we do SFT then DPO then safety," but *why* these are separate, in this order.
3. **At least one risk-to-gate traceability row per OWASP ML Top 10 category** (student may compact).
4. **Mapping to NIST AI RMF functions** on at least the four gates most clearly owned by GOVERN / MAP / MEASURE / MANAGE.
5. **Named artifacts** per stage (Markdown files, config files, reports). Vague "some report" is not enough.
6. **A rollback plan** attached to Stage 7 — the rollback target exists before release, not after an incident.

## 4. Rubric

10 points total.

| Dimension | Pts | Pass criterion |
|---|---|---|
| Stage coverage | 2 | All nine stages (0–8) named, each with input + artifact + gate |
| Ordering rationale | 2 | Defends SFT-before-PO; safety-as-separate; eval-after-weights-change; rollback-planned-upfront |
| OWASP ML traceability | 2 | Every OWASP ML Top 10 category mapped to at least one gate |
| NIST AI RMF alignment | 1 | Governance function labeled on each gate, using GOVERN/MAP/MEASURE/MANAGE |
| MITRE ATLAS references | 1 | At least two ATLAS techniques cited where relevant (supply chain, model evasion, model theft) |
| Artifact list | 1 | Named files per stage, not vague descriptions |
| Rollback plan | 1 | Rollback target and trigger defined at Stage 7 (before incident) |

Pass threshold: 7/10, with "stage coverage" ≥ 1 and "OWASP ML traceability" ≥ 1.

## 5. Common mistakes

- **Treating the map as a diagram only.** The point is the gates and the artifacts, not the arrows.
- **Rolling data curation into SFT.** Separating them is what makes decontamination auditable and what makes a data-poisoning review possible [OWASP-ML ML02].
- **Running preference optimization without SFT.** Destabilizes training; not what published recipes do [IGP22, DPO23].
- **Folding safety tuning into SFT or into preference optimization.** You lose the ability to measure over-refusal regression as a distinct release gate.
- **Using the development eval set as the release gate.** A model that trained against evals cannot block its own release [OWASP-ML ML04/ML08].
- **Treating quantization as "safe by construction."** Low-bit quantization shifts the risk surface; a parity report is a release gate [L3].
- **No defined rollback target.** Release without a rollback target violates the NIST AI RMF MANAGE function [NIST-AI-RMF].
- **Invented metrics.** The map describes gates and artifacts, not fictional incident rates. Any number that leaves this file must come from a cited source.
- **One-off citations without category.** Cite OWASP ML categories by their ML0N id and ATLAS techniques by their AML.T#### id so a reviewer can look them up.

## 6. References

- [OWASP-ML] OWASP Machine Learning Security Top 10 (2023+). <https://owasp.org/www-project-machine-learning-security-top-10/>
- [ATLAS] MITRE ATLAS — Adversarial Threat Landscape for Artificial-Intelligence Systems. <https://atlas.mitre.org/>
- [NIST-AI-RMF] NIST AI Risk Management Framework 1.0, NIST AI 100-1, 2023. <https://www.nist.gov/itl/ai-risk-management-framework>
- [IGP22] Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022. <https://arxiv.org/abs/2203.02155>
- [DPO23] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023. <https://arxiv.org/abs/2305.18290>
- [L3] Meta, "The Llama 3 Herd of Models," 2024. <https://arxiv.org/abs/2407.21783>
- HuggingFace TRL (SFT, DPO, ORPO reference trainers). <https://huggingface.co/docs/trl>
- HuggingFace Model Cards guide. <https://huggingface.co/docs/hub/model-cards>
