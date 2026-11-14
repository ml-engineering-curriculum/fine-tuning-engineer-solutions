# mod-105 · exercise-04 — DPO Run and Comparison to PPO — Solution

## 1. Solution overview

The deliverable is a DPO run on the same base and the same preference pairs as exercise-02/03, plus a side-by-side comparison with the PPO policy from exercise-03 on the same eval set. A reviewer should be able to read `train_dpo.py`, `dpo_config.yaml`, `comparison_report.md`, and the saved adapter, and sign off on which of the two (PPO or DPO) to ship for this particular workload — with an argument that goes beyond "DPO is simpler".

Three ideas carry the exercise:

- **DPO is RLHF without the RL.** The DPO loss is an algebraic rearrangement of the RLHF optimum under a Bradley–Terry reward model: the policy itself is the reward model, implicit in `log π_θ / π_ref`. No rollouts, no value head, no PPO clip, no reward model to train [DPO23].
- **"Simpler to train" does not mean "always better".** DPO is stable, deterministic, off-policy. PPO is unstable, stochastic, on-policy. They trade different failure modes: DPO over-fits to the preference pair distribution; PPO reward-hacks the explicit RM. The comparison report must capture this trade.
- **The comparison must be apples-to-apples or it is noise.** Same base, same SFT starting point, same preference pairs (or the same reward model, derived from those pairs), same eval set, same judge. Any variation in these confounds "DPO vs. PPO" with "data vs. data".

This walkthrough uses `trl.DPOTrainer` [TRL] on a LoRA-adapted `meta-llama/Meta-Llama-3.1-8B-Instruct` policy, the same SFT base as reference, and the `prefs-v1` train split from exercise-01. All numbers are illustrative; the methodology and acceptance structure are what to grade.

## 2. Worked implementation

### 2.1 `dpo_config.yaml`

```yaml
# dpo_config.yaml
run_name: dpo-v1
seed: 0

model:
  policy_base: meta-llama/Meta-Llama-3.1-8B-Instruct    # SFT from mod-103
  reference:   meta-llama/Meta-Llama-3.1-8B-Instruct    # Frozen
peft:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

data:
  pairs_train: ../exercise-01-preference-pair-dataset-construction/out/pairs_train.jsonl
  pairs_val:   ../exercise-01-preference-pair-dataset-construction/out/pairs_val.jsonl
  dimension_filter: helpful           # Train helpfulness-DPO; harmlessness-DPO is a separate run (mod-106).
  max_prompt_tokens: 512
  max_chosen_tokens: 384
  max_rejected_tokens: 384

dpo:
  beta: 0.1                           # Temperature on the implicit reward.
  loss_type: sigmoid                  # Classic DPO loss. "ipo" and "kto" variants in ex-05.
  label_smoothing: 0.0

training:
  batch_size: 4
  grad_accum: 8                       # Effective batch 32
  epochs: 2
  learning_rate: 5.0e-6
  lr_scheduler: cosine
  warmup_ratio: 0.03
  weight_decay: 0.0
  grad_clip: 1.0
  save_every: 500
  eval_every: 250
  logging_steps: 10

eval:
  held_out_prompts: eval_prompts.jsonl
  n_pairwise_judges: 3
```

### 2.2 `train_dpo.py` — reference trainer

```python
# train_dpo.py — illustrative skeleton; trimmed for brevity, not correctness.
from __future__ import annotations
import json, pathlib, yaml, torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

CFG = yaml.safe_load(open("dpo_config.yaml"))
torch.manual_seed(CFG["seed"])

tok = AutoTokenizer.from_pretrained(CFG["model"]["policy_base"])
tok.pad_token = tok.eos_token

# --- Load pairs and filter to one dimension at a time ----------------------
def _load(path: str, dim: str) -> Dataset:
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["preference_dimension"] == dim]
    # DPO expects three fields: prompt, chosen, rejected.
    data = [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in rows]
    return Dataset.from_list(data)


train_ds = _load(CFG["data"]["pairs_train"], CFG["data"]["dimension_filter"])
val_ds   = _load(CFG["data"]["pairs_val"],   CFG["data"]["dimension_filter"])

# --- Policy (LoRA-adapted) + frozen reference ------------------------------
policy = AutoModelForCausalLM.from_pretrained(
    CFG["model"]["policy_base"], torch_dtype=torch.bfloat16,
)
reference = AutoModelForCausalLM.from_pretrained(
    CFG["model"]["reference"], torch_dtype=torch.bfloat16,
)
for p in reference.parameters(): p.requires_grad_(False)

# --- DPO config ------------------------------------------------------------
dpo_cfg = DPOConfig(
    output_dir=f"dpo_out/{CFG['run_name']}",
    per_device_train_batch_size=CFG["training"]["batch_size"],
    gradient_accumulation_steps=CFG["training"]["grad_accum"],
    num_train_epochs=CFG["training"]["epochs"],
    learning_rate=CFG["training"]["learning_rate"],
    lr_scheduler_type=CFG["training"]["lr_scheduler"],
    warmup_ratio=CFG["training"]["warmup_ratio"],
    weight_decay=CFG["training"]["weight_decay"],
    max_grad_norm=CFG["training"]["grad_clip"],
    logging_steps=CFG["training"]["logging_steps"],
    save_steps=CFG["training"]["save_every"],
    eval_strategy="steps",
    eval_steps=CFG["training"]["eval_every"],
    beta=CFG["dpo"]["beta"],
    loss_type=CFG["dpo"]["loss_type"],
    label_smoothing=CFG["dpo"]["label_smoothing"],
    max_length=1024,
    max_prompt_length=CFG["data"]["max_prompt_tokens"],
    bf16=True,
    seed=CFG["seed"],
    report_to=["none"],
)

trainer = DPOTrainer(
    model=policy,
    ref_model=reference,
    args=dpo_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tok,
    peft_config=LoraConfig(
        r=CFG["peft"]["r"],
        lora_alpha=CFG["peft"]["alpha"],
        lora_dropout=CFG["peft"]["dropout"],
        target_modules=CFG["peft"]["target_modules"],
        task_type="CAUSAL_LM",
    ),
)
trainer.train()
trainer.save_model(f"dpo_out/{CFG['run_name']}/final")
```

Notes on the reference:

- **The DPO loss**, in one line: `L = −E[(x, y_c, y_r)] log σ( β ( log π_θ(y_c | x) − log π_ref(y_c | x) − log π_θ(y_r | x) + log π_ref(y_r | x) ) )` [DPO23]. The reference-anchored ratio is the implicit reward; the sigmoid-of-difference is Bradley–Terry's likelihood. There is no reward model, no sampling.
- **β controls the temperature on the implicit reward.** `β ≈ 0.1` is the paper's default for Pythia / Llama bases; smaller β lets the policy diverge further from reference (looser regularization), larger β keeps it closer. Grid-search over `{0.05, 0.1, 0.2, 0.5}` if you don't know your pool.
- **Reference model must be frozen and must be the SFT policy.** This is non-negotiable; DPO's implicit reward is defined relative to reference. A reference that is, say, the base pretrained model (not the SFT) produces a *different* implicit reward and the optimization drifts.
- **One dimension at a time.** `prefs-v1` carries helpful and harmless pairs. Training both under one DPO head yields a policy that optimizes a Bradley–Terry model over a mix of objectives; harder to interpret than two DPO passes or a composite loss. Exercise-05 explores variants.
- **No value head, no rollouts, no reward model.** This is the "simpler" bit of DPO; it is also why DPO can over-fit the pairs: there is no on-policy correction.
- **LoRA on the policy, no LoRA on the reference.** Same convention as PPO; same reason.

### 2.3 Monitoring — what "healthy" looks like

DPO's logs are different from PPO's. The key TRL metrics are:

| Metric | Healthy start | Healthy end (~epoch 2) | Unhealthy signature |
|---|---|---|---|
| `train/loss` | ~0.693 (log 2; both logits equal) | 0.35–0.55 | Flat at 0.69 (nothing learned) or ~0 (over-fit on tiny pool) |
| `rewards/accuracies` | ~0.50 | 0.70–0.85 | ≥ 0.95 on train while val plateaus at 0.60 → over-fit |
| `rewards/margins` | ~0 | +0.6 to +1.5 | Negative (chosen ranked below rejected: bug or data issue) |
| `rewards/chosen` | ~0 | +0.3 to +1.0 | Large negative while margins stay positive → policy pulling AWAY from chosen (check reference) |
| `rewards/rejected` | ~0 | −0.3 to −0.9 | Positive → policy prefers rejected on training pool |
| `logps/chosen`, `logps/rejected` | both decline (natural with PEFT) | chosen decline is slower than rejected | Both climb → reference/policy mismatch |
| `eval/rewards/accuracies` | ~0.50 | tracks train, lag ≤ 10 pts | Larger lag → over-fit; see below |

### 2.4 `comparison_report.md` — target shape

```markdown
# PPO vs. DPO Comparison — `ppo-v1` vs. `dpo-v1`

## Apples-to-apples setup
- Same base: Llama-3.1-8B-Instruct SFT (mod-103).
- Same preference source: prefs-v1 helpful split (ex-01).
  - PPO uses the rm-v1 reward model trained on these pairs (ex-02).
  - DPO uses the pairs directly, no explicit RM.
- Same reference policy for both.
- Same held-out eval set: 500 prompts (SFT-eval-decontaminated).
- Same judge protocol: 2-template model-as-judge, order-randomized, 3 votes.
- Same hardware (2× A100 80 GB).

## Resource / wall-clock

| | PPO-v1 (ex-03) | DPO-v1 (ex-04) |
|---|---|---|
| Wall clock | ~14 h | ~2.5 h |
| Peak VRAM | ~72 GB (policy + ref + RM) | ~60 GB (policy + ref) |
| Moving pieces | Reward model, value head, PPO clip, adaptive KL, length penalty | β, LR, LoRA config |
| Hyperparameters to tune | ~8 | ~3 |
| Determinism | Stochastic (rollouts) | Deterministic (fixed data order) |

## Quality numbers

| Metric | SFT (baseline) | PPO-v1 | DPO-v1 |
|---|---:|---:|---:|
| rm-v1 score mean on 500 held-out prompts | +0.04 | +1.33 | +1.21 |
| Judge win rate vs. SFT | — | 61% | 58% |
| Judge win rate vs. each other | — | 46% | 43% (11% tie) |
| Mean response length (tokens) | 118 | 198 | 171 |
| Distinct-2 on rollouts | 0.47 | 0.41 | 0.43 |
| Safety refusal correctness (120 red-team prompts) | 76% | 71% | 74% |
| Val pair accuracy (DPO's "rewards/accuracies") | n/a | n/a | 0.78 |
| Val pair accuracy under rm-v1 (as external judge) | n/a | 0.76 | 0.74 |

### Observations

- **PPO wins on RM score, DPO wins on length and safety.** PPO explicitly
  chased rm-v1's signal; DPO fits the pair distribution directly, which
  includes safety-dimension pairs' influence on helpfulness pairs.
- **Judge side-by-side is close to a tie.** 46% / 43% / 11% means the two
  policies are indistinguishable to a judge on roughly the same fraction
  of prompts they jointly beat SFT on. Humans and model-judges don't see
  a 1.33 vs. 1.21 difference in rm-v1 score as a win rate difference.
- **Wall clock is 5.6× faster for DPO.** For the same team iterating on
  preference data (ex-01 v2, v3...), DPO is a dramatically shorter inner
  loop.

## Where DPO is worse

- **Over-fitting on narrow pools.** When we ran DPO on only the 500-pair
  "safety refusal" slice from ex-01, val pair accuracy climbed to 0.91 but
  the judge win rate vs. SFT dropped to 51% — the policy learned the exact
  refusal strings in the pool and nothing more general.
- **No on-policy correction.** DPO does not see what the current policy
  would sample, so pairs that drift out-of-distribution from π_θ's actual
  outputs provide weak signal. PPO's rollouts don't have this problem.
- **Harder to steer with rewards you haven't labeled.** Length, format,
  tone — anything you can measure at rollout time, PPO can shape through
  the reward. DPO has to encode it in the pair labels.

## Where PPO is worse

- **Reward hacking.** We saw a +0.34 length-vs-reward correlation on PPO
  rollouts (ex-03) and had to add a length penalty to keep length under
  control. DPO showed +0.12 — the pair distribution itself is less
  length-biased when you never ask the RM for a continuous score.
- **Training instability.** PPO required 3 restarts in pilot runs to tune
  init_kl_coef / target_kl. DPO ran once at β=0.1 and shipped.
- **Harder to reproduce.** PPO rollout order is stochastic; on the same
  seed and hardware, final policy differs in the 4th decimal of eval
  metrics. DPO is deterministic to bitwise equality on the same hardware.

## Recommendation (for this project)

- **Ship DPO for helpful-only.** 5.6× faster training, comparable judge win
  rate, better length and safety behavior, deterministic rebuild.
- **Retain PPO for scenarios where**:
  (a) we have a reward signal we did not label directly (e.g. execution
      reward for code, unit-test pass/fail, user telemetry), or
  (b) the preference pool is narrow and DPO over-fits.
- **Do both, on the same data, as a canary.** When the DPO vs. PPO judge
  win rate drops below 40/40 on a fixed canary set, something upstream has
  shifted and we have an early-warning signal.

## Known risks

- DPO's over-fit mode is quieter than PPO's reward hack; the "rewards/accuracies"
  on train looks great while val stagnates. The acceptance criterion requires
  a train↔val gap ≤ 10 pts.
- Both methods inherit the ex-01 length and style biases. Neither is a
  mitigation for data-level issues.
- Switching between PPO and DPO mid-iteration without re-baselining the judge
  is confounded: judge version drift is a real signal.
```

### 2.5 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Use the same preference pairs as PPO's RM training | The only honest comparison | Fresh pairs for DPO | Can't attribute differences to the method vs. the data |
| β = 0.1 | DPO paper's default on Llama-class bases | β = 0.5 | Tighter reference regularization; slower learning on this pool |
| Loss type `sigmoid` | Classic DPO (Bradley–Terry) | `ipo` (removes BT assumption) | ipo is the right choice when BT doesn't hold; exercised in ex-05 |
| Helpful-only dimension filter | One objective at a time | Train on mixed dimensions | Policy's harmlessness is a side effect, not a target; by-design |
| Same LoRA config as PPO | Removes "adapter capacity" as a confound | Full-weight DPO | Would be apples-to-apples with a full-weight PPO only |
| Reference = SFT (frozen, no LoRA) | DPO's implicit reward is defined relative to reference | Reference = base pretrained | Different reward semantics; results aren't comparable to PPO's |
| Deterministic data order | Reproducibility; DPO's only source of stochasticity is dropout | Shuffled with Generator seeded | Still reproducible with `Generator().manual_seed(...)`; small gain |
| Judge win rate as the primary A/B metric | RM score is self-referential for PPO | RM score only | PPO would "win" by construction |

## 3. Validation steps

1. **Step-0 logp equivalence.** With LoRA zero-init, policy and reference produce identical log-probs on the same input; `rewards/chosen` and `rewards/rejected` at step 0 are both ~0, margins ~0. If not, LoRA is non-zero-init or reference is a different checkpoint.
2. **`train/loss` crosses `log 2 ≈ 0.693` from above in the first ~50 steps.** Staying at or above this means the margins aren't growing; check β, LR, and that chosen/rejected aren't swapped in the data loader.
3. **`rewards/accuracies` on val ≥ 0.65 by end of epoch 2.** Below this DPO has not learned a useful preference. Not comparable to PPO's "reward number" but comparable to ex-02's RM pair accuracy on the same split.
4. **Train vs. val `rewards/accuracies` gap ≤ 10 points.** Larger gap = DPO over-fit; the pool is too small or the pairs are too narrow.
5. **Judge win rate vs. SFT ≥ 55% on held-out.** Below this the policy has not learned a human-legible improvement.
6. **Length-vs-reward Pearson ≤ +0.40** on DPO rollouts, same instrument as ex-02/03. Catches length-bias drift.
7. **Safety-slice regression ≤ 10 points vs. SFT.** Same bar as PPO.
8. **Deterministic rebuild on seed 0.** Same hardware, same PRNG → bitwise-identical val metrics to the 3rd decimal (DPO's determinism is a feature; exercise it).

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **DPO loss & reference discipline** | Reference unfrozen, or missing β, or loss computed as MSE | Reference frozen but non-SFT | Reference = frozen SFT, β explicit, sigmoid (BT) loss, logp decomposition verified |
| **Data parity with PPO** | Different pairs or different dimension mix | Same pairs, but different split boundaries | Same pairs, same split, same dimension filter as the PPO's RM training |
| **Over-fit / regularization check** | No val pair-accuracy reported | Val reported but no train/val gap check | Train/val gap monitored, ≤ 10 pts, β chosen with a short sweep |
| **Side-by-side comparison** | Only standalone DPO numbers | DPO vs. SFT only | DPO vs. SFT vs. PPO on same prompts, same judge, same safety slice |
| **Recommendation & risks** | "DPO is simpler, so we ship it" with no comparison | Comparison but no explicit ship / rollback decision | Explicit per-scenario recommendation, risks listed, canary A/B defined |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- **Running DPO on a different preference pool than PPO's RM was trained on.** Any quality difference is now confounded with the pool. Use the same `pairs_train.jsonl`.
- **Mixing helpful and harmless pairs into one DPO pass.** The implicit reward now encodes a mix; the policy's behavior on either dimension is harder to steer and interpret. Train two DPOs or sequence with mod-106.
- **β chosen "because the paper said 0.1".** β interacts with your SFT's log-prob calibration. A short β sweep over `{0.05, 0.1, 0.2, 0.5}` on val is non-optional.
- **Reference not frozen.** The implicit reward collapses as the reference moves; DPO's optimum is no longer well-defined.
- **Over-fit un-caught because val `rewards/accuracies` was never computed.** A DPO that scores 0.95 on train and 0.60 on val is a worse policy than one that scores 0.78/0.74 — but the first *looks* better in logs.
- **Comparing wall clock as if it were the whole story.** DPO is 5× faster, yes. If the wrong signal is faster to train on, it's still the wrong signal.
- **Shipping DPO on RM-score alone.** Same trap as PPO, inverted: DPO doesn't optimize rm-v1 directly, so its rm-v1 score might be lower, even when the judge rates it tied with PPO. Judge > RM score for A/B.
- **Treating "DPO vs. PPO" as an engineering religion.** They are different estimators of the same objective; the right call depends on the pool, the eval, and whether you have a reward signal beyond preferences.
- **Swapped chosen/rejected in the data loader.** `rewards/margins` goes negative and the policy learns the wrong direction. Always spot-check 5 rows after loading.
- **No decontam check against eval prompts.** Same discipline as ex-02/03 — preference prompts that are also eval prompts invalidate the eval.

## 6. References

- [DPO23] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023. <https://arxiv.org/abs/2305.18290>
- [IPO23] Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO loss), 2023. <https://arxiv.org/abs/2310.12036>
- [InstructGPT22] Ouyang et al., 2022. <https://arxiv.org/abs/2203.02155>
- [PPO17] Schulman et al., 2017. <https://arxiv.org/abs/1707.06347>
- [TRL] Hugging Face TRL — `DPOTrainer`. <https://github.com/huggingface/trl> — `trl/trainer/dpo_trainer.py` is the authoritative reference.
- `peft` (LoRA / QLoRA). <https://github.com/huggingface/peft>
- Transformers. <https://github.com/huggingface/transformers>
- Meta-Llama-3.1-8B-Instruct model card. <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>
- OWASP Machine Learning Security Top 10 (ML-02 Data Poisoning via preference pairs — same pool feeds both PPO and DPO, so the attack surface is identical; ML-06 Supply Chain). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — misalignment via corrupted preferences ("Resource Development"). <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Measure/Manage; A/B comparisons and canary sets are Measure obligations; the recommendation is a Govern artifact). <https://www.nist.gov/itl/ai-risk-management-framework>
