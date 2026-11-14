# mod-105 · exercise-05 — ORPO / KTO / SimPO / GRPO Survey Experiment — Solution

## 1. Solution overview

The deliverable is a survey matrix across four post-DPO preference-optimization methods plus a minimal experiment that runs each on the same base and reports the same metrics, so a reader can pick a default by scenario. It is explicitly *not* a single-winner shootout; the point is to show when each method's inductive bias matches the problem. A reviewer should be able to read `methods_matrix.md`, `train_*.py`, `survey_report.md`, and the saved checkpoints, and justify a choice without re-reading the papers.

Three ideas carry the exercise:

- **Each method exists because DPO has a specific gap.** ORPO folds the SFT loss into the preference loss, so you can skip the separate SFT stage entirely [ORPO24]. KTO drops the paired-preference assumption and trains on binary labels (desirable / undesirable) via a prospect-theoretic loss [KTO24]. SimPO removes the reference policy and uses a length-normalized implicit reward [SimPO24]. GRPO replaces PPO's value head with a group-relative baseline computed from multiple samples per prompt [DeepSeekMath24]. Picking one is picking which gap matches your data.
- **The comparison matrix is the deliverable. The experiment is the receipt.** The matrix encodes data shape, compute cost, and known failure modes. The experiment confirms the matrix reflects reality on your pool.
- **Hype decays faster than the matrix.** Every six months a new method claims +N on a small eval; the matrix stays anchored on reproducible axes (what inputs it needs, what wall clock it costs, how it fails, what library implements it).

This walkthrough uses `trl.*Trainer` for ORPO, KTO, and SimPO (via DPOTrainer with `loss_type=simpo`), and a short custom GRPO loop built on top of a policy + sampling + advantage computation. All numbers are illustrative; the methodology is what to grade.

## 2. Worked implementation

### 2.1 `methods_matrix.md` — the main artifact

```markdown
# Post-DPO Preference Optimization — Methods Matrix

Reference base for all numbers: a 7–8B decoder-only LLM, LoRA r=16 on qkvo,
2× A100 80 GB, 5,000-pair preference pool (prefs-v1 from ex-01) or 10,000
single-label rows for KTO. "Baseline" = SFT from mod-103.

## A. Input shape (what the method consumes)

| Method | Inputs needed | Reference policy? | Reward model? | Single-label or pairs? |
|---|---|---|---|---|
| **DPO** (ex-04) | `(prompt, chosen, rejected)` triples | Yes (frozen SFT) | No | Pairs |
| **ORPO** [ORPO24] | `(prompt, chosen, rejected)` triples | No | No | Pairs |
| **KTO** [KTO24] | `(prompt, response, label ∈ {good, bad})` rows | Yes (frozen SFT) | No | Single labels |
| **SimPO** [SimPO24] | `(prompt, chosen, rejected)` triples | No | No | Pairs |
| **GRPO** [DeepSeekMath24] | `(prompt, [N sampled responses], reward(y_i))` groups | Yes (frozen SFT, for KL) | Yes (or a programmatic reward) | Groups of N (default N=4) |

## B. Loss sketch (one line each)

- **DPO**: `−log σ( β · ( Δ log π_θ − Δ log π_ref ) )`, where Δ = chosen − rejected.
- **ORPO**: `L_SFT(y_c | x) + λ · L_OR(x, y_c, y_r)` with `L_OR = −log σ( log(odds_θ(y_c) / odds_θ(y_r)) )`. No reference policy; SFT term carries the base-distribution anchor.
- **KTO**: `−E_desirable[ v( β · (log π_θ − log π_ref)) ] + E_undesirable[ v( ... ) ]`, where `v(·)` is a Kahneman–Tversky value function that is concave in gains and convex in losses — captures loss-aversion.
- **SimPO**: `−log σ( β · ( (1/|y_c|) log π_θ(y_c | x) − (1/|y_r|) log π_θ(y_r | x) ) − γ )`. Length-normalized implicit reward; `γ` is a target reward margin; no reference policy at all.
- **GRPO**: `−E[ Σ_t  min( r_t · A_grp, clip(r_t, 1−ε, 1+ε) · A_grp ) ] + β · KL(π_θ || π_ref)` where `A_grp = (R_i − mean_group(R)) / std_group(R)`. PPO shape without a value head.

## C. What each method solves relative to DPO

| Gap in DPO | Method | Mechanism |
|---|---|---|
| Needs a separate SFT run upstream | **ORPO** | Fuses SFT + preference; one stage, pair data |
| Needs expensive paired comparisons | **KTO** | Binary good/bad labels are often cheaper to collect |
| Policy may drift from reference in a way β doesn't control | **KTO** | Prospect-theoretic loss is more robust to label noise |
| Implicit reward is length-biased (DPO's known flaw) | **SimPO** | Length-normalize the implicit reward |
| Reference policy is expensive to keep in memory | **SimPO** | No reference at all; use a target margin γ |
| On-policy signal not available because no RM | **GRPO** | Multi-sample on-policy; group-relative baseline replaces the value head of PPO |
| PPO value head is unstable | **GRPO** | Variance reduction via group-relative advantages |

## D. Compute and operational cost (relative, same base and same pool)

| | DPO | ORPO | KTO | SimPO | GRPO |
|---|---|---|---|---|---|
| Needs SFT model | Yes | **No** (does SFT + preference jointly) | Yes | Yes | Yes |
| Needs frozen reference in memory | Yes | **No** | Yes | **No** | Yes |
| Needs reward model | No | No | No | No | Usually (or programmatic reward) |
| Sampling at train time | **No** | **No** | **No** | **No** | **Yes (N per prompt)** |
| Wall clock (2× A100, 5k pairs, 2 epochs) | ~2.5 h | ~3 h | ~3 h (10k rows) | ~2.5 h | ~5 h (per-step N=4 sampling) |
| Peak VRAM | ~60 GB | ~45 GB | ~60 GB | ~45 GB | ~70 GB |
| Deterministic rebuild | Yes | Yes | Yes | Yes | No (sampling) |
| Library | trl.DPOTrainer | trl.ORPOTrainer | trl.KTOTrainer | trl.DPOTrainer(loss_type=simpo) | trl.GRPOTrainer (0.9+) or custom |

## E. Known failure modes

| Method | Failure mode | Signal |
|---|---|---|
| DPO | Over-fits on narrow pools; length-biased | Train-val `rewards/accuracies` gap, length Pearson |
| ORPO | Single-stage ambition clashes with heterogeneous data (SFT on noisy + preference on clean breaks assumption) | SFT loss plateaus high while OR loss falls |
| KTO | Mislabeled binary labels blow up the loss asymmetrically (loss aversion amplifies label noise) | Loss spikes on rare "undesirable" examples |
| SimPO | γ wrong for your SFT → policy either never learns or diverges with no reference to pull it back | `rewards/margins` either stuck at 0 or runaway positive |
| GRPO | Group size N too small (noisy baseline) or rewards degenerate (all identical) | Advantage std ~0; gradient vanishes |

## F. When to use what (decision tree)

1. **Do you have a reward signal (not just preferences)?**
   - Yes (e.g. unit-test pass/fail, user telemetry, execution reward):
     **PPO (ex-03)** or **GRPO**. GRPO if you want to drop the value head
     and are OK with N× sampling cost.
   - No: go to (2).
2. **Are your labels pairs or binary?**
   - Pairs: go to (3).
   - Binary (good / bad): **KTO**.
3. **Do you have a separate SFT stage?**
   - Yes: **DPO (ex-04)** is the safe default.
   - No, and you want to collapse into one stage: **ORPO**.
4. **Are DPO's length bias or the reference-in-memory cost blocking you?**
   - Yes: **SimPO** (consider; it is the newest here, less mileage).
   - No: stay on DPO.
```

### 2.2 `train_orpo.py` — shape

```python
# train_orpo.py — one-stage SFT + preference via trl.ORPOTrainer.
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import Dataset
import json

tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tok.pad_token = tok.eos_token

# Load the SAME pairs as DPO used in ex-04, helpful only.
rows = [json.loads(l) for l in open("../ex-01/out/pairs_train.jsonl")
        if json.loads(l)["preference_dimension"] == "helpful"]
ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"],
                         "rejected": r["rejected"]} for r in rows])

policy = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B", torch_dtype="bfloat16"   # base, NOT SFT'd
)

cfg = ORPOConfig(
    output_dir="orpo_out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=8e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    beta=0.1,                          # ORPO's λ, "odds-ratio weight"
    max_length=1024,
    max_prompt_length=512,
    bf16=True,
    seed=0,
    report_to=["none"],
)

trainer = ORPOTrainer(
    model=policy,
    args=cfg,
    train_dataset=ds,
    tokenizer=tok,
    peft_config=LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ),
)
trainer.train()
```

Notes:
- **No reference model argument.** ORPO doesn't use one; the SFT term on `chosen` is the base-distribution anchor.
- **Starts from the *base*, not an SFT.** This is ORPO's whole point: SFT + preference in one pass.

### 2.3 `train_kto.py` — shape

```python
# train_kto.py — binary-label preference training.
from trl import KTOConfig, KTOTrainer
from datasets import Dataset
import json

# Derive a binary dataset from the paired dataset: each row is split into
# two single-label rows. (chosen → label=True), (rejected → label=False).
rows = [json.loads(l) for l in open("../ex-01/out/pairs_train.jsonl")
        if json.loads(l)["preference_dimension"] == "helpful"]
kto_rows = []
for r in rows:
    kto_rows.append({"prompt": r["prompt"], "completion": r["chosen"], "label": True})
    kto_rows.append({"prompt": r["prompt"], "completion": r["rejected"], "label": False})
ds = Dataset.from_list(kto_rows)

cfg = KTOConfig(
    output_dir="kto_out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=5e-6,
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
    max_length=1024,
    max_prompt_length=512,
    bf16=True,
    seed=0,
    report_to=["none"],
)
# trainer = KTOTrainer(model=sft, ref_model=sft_frozen, args=cfg, train_dataset=ds, tokenizer=tok, peft_config=...)
# trainer.train()
```

Notes:
- **Binary labels, not pairs.** We synthesized binary from pairs for the survey. In a native KTO setup you would collect thumbs-up / thumbs-down telemetry directly; the method is designed to use that cheaper signal.
- **`desirable_weight` / `undesirable_weight`.** KTO's loss aversion means `undesirable_weight = 1` is already ~2× the pull of an equally-sized desirable example. Set `undesirable_weight < 1` if your bad examples are noisier than your good ones.

### 2.4 `train_simpo.py` — shape

```python
# train_simpo.py — length-normalized reference-free DPO variant via trl DPOTrainer.
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
import json

rows = [json.loads(l) for l in open("../ex-01/out/pairs_train.jsonl")
        if json.loads(l)["preference_dimension"] == "helpful"]
ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"],
                         "rejected": r["rejected"]} for r in rows])

cfg = DPOConfig(
    output_dir="simpo_out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    beta=2.0,                          # SimPO uses larger β than DPO (10–20×); paper default 2.0 on Llama-3-8B.
    loss_type="simpo",
    simpo_gamma=1.0,                   # target reward margin γ.
    max_length=1024,
    max_prompt_length=512,
    bf16=True,
    seed=0,
    report_to=["none"],
)
# trainer = DPOTrainer(model=sft, args=cfg, train_dataset=ds, tokenizer=tok, peft_config=...)
# No ref_model=... passed; SimPO is reference-free. Trainer handles this.
```

Notes:
- **No `ref_model` argument.** SimPO's implicit reward is `(1/|y|) · log π_θ(y | x)`; there is no reference log-prob subtracted.
- **β is ~10–20× larger than DPO's.** Different normalization (length-normed log-probs are smaller in absolute value), so β compensates.
- **`simpo_gamma`.** Target margin. Grid-search in `{0.3, 0.5, 1.0, 1.5}` on a short val slice; the paper reports γ≈1.0 for 7B bases [SimPO24].

### 2.5 `train_grpo.py` — shape

```python
# train_grpo.py — group-relative policy optimization. Uses the rm-v1 reward
# model from ex-02 as the reward function; could be any programmatic reward.
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import Dataset
import json, torch

tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tok.pad_token = tok.eos_token

# Load unique prompts; GRPO samples from the current policy at train time.
prompts = [json.loads(l)["prompt"]
           for l in open("../ex-03/prompts.jsonl")]
ds = Dataset.from_list([{"prompt": p} for p in prompts])

# Reward function: wrap the ex-02 RM (or any scalar reward).
rm = _load_rm("../ex-02/rm_out/adapter", "../ex-02/rm_out/score_head.pt").cuda()

def reward_fn(prompts, completions, **_):
    scores = rm.score_responses(prompts, completions)
    return scores.tolist()

cfg = GRPOConfig(
    output_dir="grpo_out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=4,                 # N, group size
    num_train_epochs=1,
    learning_rate=1e-5,
    beta=0.04,                         # KL coefficient
    temperature=1.0,
    max_prompt_length=512,
    max_completion_length=384,
    bf16=True,
    seed=0,
    report_to=["none"],
)
# trainer = GRPOTrainer(
#     model=sft, reward_funcs=[reward_fn], args=cfg,
#     train_dataset=ds, tokenizer=tok,
#     peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"],
#                            task_type="CAUSAL_LM"),
# )
# trainer.train()
```

Notes:
- **`num_generations = N`.** The group size. The baseline is `R_i − mean_group(R)`, standardized by `std_group(R)`. With N=1 GRPO degenerates to REINFORCE with no baseline.
- **No value head.** GRPO's whole point is that the group baseline replaces PPO's value head [DeepSeekMath24].
- **Reward function is pluggable.** For code or math, use an executable reward (unit tests, grader). For preferences, use the RM from ex-02. Both are "programmatic rewards" to GRPO.
- **Runtime cost is ~N× the sampling cost per step.** With N=4 and 512-token contexts the 5× DPO wall clock we observed matches expectation.

### 2.6 `survey_report.md` — target shape

```markdown
# Preference Optimization Survey — DPO vs. ORPO vs. KTO vs. SimPO vs. GRPO

## Shared setup
- Base: meta-llama/Meta-Llama-3.1-8B (or -Instruct for methods needing SFT).
- LoRA r=16, qkvo, bf16.
- Preference pool: prefs-v1 helpful split (5k pairs), same val/test split as ex-02/04.
- KTO binary data: derived from the same pool (2× rows).
- GRPO reward: rm-v1 from ex-02.
- Eval: 500 held-out prompts, decontaminated.
- Judge: 2-template model-as-judge, order-randomized, 3 votes.
- Hardware: 2× A100 80 GB.

## Results

| Method | Starts from | Wall clock | VRAM peak | rm-v1 score mean (test) | Judge win vs. SFT | Length (mean) | Val pair acc |
|---|---|---:|---:|---:|---:|---:|---:|
| SFT (baseline) | base | 4 h (mod-103) | 70 GB | +0.04 | — | 118 | n/a |
| DPO (ex-04) | SFT | 2.5 h | 60 GB | +1.21 | 58% | 171 | 0.78 |
| ORPO | base (one-stage) | 3.0 h | 45 GB | +0.98 | 54% | 164 | 0.73 |
| KTO | SFT | 3.0 h | 60 GB | +1.08 | 56% | 158 | 0.71 (binary) |
| SimPO | SFT | 2.5 h | 45 GB | +1.14 | 57% | 149 | 0.74 |
| GRPO | SFT | 5.0 h | 70 GB | +1.29 | 60% | 183 | n/a |
| PPO (ex-03) | SFT | 14 h | 72 GB | +1.33 | 61% | 198 | n/a |

### Interpretation

- **SFT → DPO is the strongest single improvement.** Everything else in the
  pack is within judge-noise of DPO and PPO.
- **ORPO is the cheapest viable option on data-constrained teams.** 3 h, 45
  GB, no separate SFT stage, no reference. Pays for it with the lowest
  judge win rate — judge noise could absorb this on different pools.
- **SimPO closes most of DPO's gap at lower memory and tighter length.**
  Mean length 149 vs. DPO's 171 — the length-normalized implicit reward
  works as advertised on this pool.
- **KTO wins no axis, costs about the same as DPO, but is the only method
  here that consumes binary telemetry natively.** For teams with thumbs
  up/down signals, that is the whole point; for pair-labeled pools, it's a
  wash.
- **GRPO closes most of PPO's gap at 3× the wall clock of DPO and ~1/3 the
  wall clock of PPO.** Its killer feature is when your reward is
  programmatic (unit tests, graders) — then you don't even need an RM.
- **PPO still wins on rm-v1 mean reward**, but it chases rm-v1
  explicitly; see ex-03's length bias and safety regression.

## Failure-mode receipts

### ORPO on heterogeneous data
When we ran ORPO on a pool that mixed clean SFT examples with noisy
preference pairs, the SFT loss plateaued at 1.8 while OR loss fell to 0.4;
the resulting policy was worse than SFT. Fix: separate the clean SFT pool
from the preference pool, or revert to DPO. Signal: `train/sft_loss`
plateau while `train/odds_ratio_loss` keeps falling.

### KTO with noisy labels
We injected 5% label noise in the binary set (flipped good↔bad on random
rows). KTO's val accuracy dropped from 0.71 to 0.58. The loss-aversion
weighting of "bad" examples made noise asymmetric. Fix: lower
`undesirable_weight` to 0.5 when labels are noisy.

### SimPO with the wrong γ
At γ=0.3 the margin never grew; at γ=2.0 the policy diverged. γ=1.0 was in
the sweet spot. A one-dimensional sweep over `{0.3, 0.5, 1.0, 1.5}` on a
250-prompt val slice is non-optional.

### GRPO with degenerate groups
On prompts where all 4 samples got similar RM scores (rare, happens on
very-safe prompts like "what is 2+2"), the advantage std → 0 and the
update was a no-op. Not catastrophic; just wasted compute. Fix: filter
groups with `std(R) < 1e-3` from the gradient step.

## Recommendation (for this project)

- **Default for new preference work: DPO.** Simplest, deterministic,
  smallest surface area to misconfigure.
- **ORPO** if you are collapsing SFT + preference into one stage to save
  a run.
- **KTO** if your data is binary telemetry.
- **SimPO** if DPO's length bias is blocking, and you can afford a short
  γ sweep.
- **GRPO** if you have a programmatic (non-preference) reward or need
  PPO-style on-policy behavior without the value head.
- **PPO** if GRPO's group variance is pathological on your reward
  (e.g. sparse unit-test pass/fail) — then the value head's bias trades
  worthwhile for its variance reduction.

## Open questions / known gaps

- SimPO on bases that are *not* already instruction-tuned: γ reports in
  the paper are for Llama-3-8B-Instruct; our one-shot result at γ=1.0 may
  not transfer to a base model. Needs a sweep at γ ∈ {0.5, 1.0, 1.5}.
- GRPO on *very* long reasoning chains (code, math with chain-of-thought):
  the group-relative baseline becomes noisy as rewards concentrate on a
  tiny fraction of rollouts. We did not test this configuration.
- The SFT regression we observed in ORPO may be tunable with a larger λ;
  we did not do a full sweep.
```

### 2.7 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Share the same pool across methods | Only way the comparison is honest | Each method on its own curated pool | Would measure data effects, not method effects |
| Same eval set and judge protocol | Judge drift dominates method drift otherwise | New judge per method | Numbers become uncomparable |
| LoRA at the same rank for all | Removes "adapter capacity" as a confound | Each method with its paper's recipe | Would measure recipe drift too |
| Report 6 metrics, not 1 | Methods differ along different axes (length, SFT need, reward need, VRAM) | Pick a single "winner" | Overclaims; next pool reverses the ranking |
| Include a known-failure receipt per method | Survey without failure modes is marketing copy | Only success numbers | Readers can't choose for their own pool |
| Keep PPO (ex-03) in the comparison | It's the reference point that motivated all four | Omit PPO | Can't argue "x is better than PPO for ..." |
| β, γ, λ grids on val slice | Default hyperparameters are per-paper, per-base; survey without sweep is misinformation | Use paper defaults as-is | Short sweeps are cheap and prevent reporting bad numbers |
| Deterministic rebuild where the method allows it | Reproducibility; part of each method's documented failure story | Skip | Lose the ability to A/B subsequent policy versions |

## 3. Validation steps

1. **Per-method step-0 sanity.** DPO/KTO/GRPO have an SFT-equivalent policy at step 0 (LoRA zero-init) → initial `rewards/margins` ≈ 0 and loss ≈ `log 2` where applicable. ORPO starts from base, so SFT loss ≈ cross-entropy of base on `chosen` (non-trivial baseline; log it). SimPO's implicit reward at step 0 is `(1/|y|) log π_θ(y | x)`, which is a constant per-(y, x) pair; `rewards/margins` ≈ 0.
2. **Per-method train/val gap monitored.** Over-fit signatures look different per method: DPO/SimPO show it in `rewards/accuracies`; ORPO shows it in SFT-vs-OR loss divergence; KTO shows it in bad-label loss asymmetry; GRPO shows it in advantage-std collapse.
3. **Length-vs-reward correlation reported for each method.** The length bias from ex-02's RM propagates differently through each method; the receipt is what separates "our RM is biased" from "our method amplifies the bias".
4. **Judge win rate against SFT ≥ 55%** on each method. Below that, the method did not produce a human-legible improvement, and the standalone pair-accuracy / mean-reward numbers are not enough to ship.
5. **Hyperparameter sweep evidence per method.** For each, show that the reported numbers are at (or near) the optimum of a short sweep; otherwise the comparison is between poorly-tuned methods.
6. **Reproducibility.** Deterministic methods (DPO, SimPO, ORPO, KTO at fixed seed and order) must reproduce exactly on the same hardware. GRPO is non-deterministic by sampling; document the seed and expect fp32-noise-level drift.
7. **Decision tree exercises match the numbers.** Walk the tree in `methods_matrix.md` with the project's constraints; the method it points at should be the one the experiment agrees with. If not, the tree needs rewriting (not the numbers).

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Method coverage & correctness** | Only two methods, or one method misconfigured (e.g. SimPO with a reference model) | All four methods, but at least one with clear misconfiguration | All four methods set up correctly per their paper, matching trl API where applicable |
| **Apples-to-apples experiment** | Different pool, judge, or eval across methods | Same pool but no reported hyperparameter sweep | Same pool, same eval, same judge, per-method short sweep documented |
| **Matrix quality (inputs/cost/failure modes)** | Only cost numbers | Inputs + cost | Inputs + cost + failure modes + per-method signals |
| **Receipts for failures** | No failure demonstrated | Reported one failure in prose | Reproduced at least one failure per method with the signal that would catch it |
| **Recommendation structure** | "X is best" with no caveats | Decision tree without constraints | Decision tree + per-scenario recommendation + open questions / known gaps |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- **Single-shootout framing.** "Which is best" is the wrong question; the pool and the reward signal pick for you. The matrix is the deliverable.
- **Different pools across methods.** The most common confound; any ranking change gets attributed to the method when it's actually the data.
- **ORPO run from a pre-SFT model.** ORPO's whole premise is one-stage SFT + preference. Starting from an SFT and running ORPO on top yields a confusing result that doesn't match the paper.
- **KTO run on paired data without respecting the binary semantics.** KTO expects uncorrelated good/bad rows, not a 1:1 mirrored pair split; the loss-aversion dynamics are different when the distribution is balanced-by-construction.
- **SimPO with a reference model.** TRL's `loss_type="simpo"` already handles reference-free math; passing `ref_model=` either errors or is silently ignored, and users who think they're regularizing with KL aren't.
- **GRPO with `num_generations=1`.** You've written a REINFORCE loop with no baseline. Expect huge variance.
- **Using defaults from the papers without a short sweep on your base.** β (DPO/SimPO), γ (SimPO), λ (ORPO), weights (KTO), N (GRPO) are pool- and base-sensitive. 2-day sweeps save 2-week debugs.
- **Reporting rm-v1 score as the ranking.** Methods that explicitly optimize rm-v1 (GRPO via reward_fn=RM, PPO) have a built-in lead on this metric. Judge win rate is a less self-referential signal.
- **Not logging per-method failure signals.** ORPO's SFT/OR loss divergence is only visible if you log both; KTO's asymmetric loss blow-up is only visible if you break the loss by label; GRPO's advantage-std collapse only shows if you log group std.
- **Treating survey numbers as benchmarks.** They are illustrative receipts on a specific pool. The matrix is portable; the numbers aren't.

## 6. References

- [ORPO24] Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model," 2024. <https://arxiv.org/abs/2403.07691>
- [KTO24] Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization," 2024. <https://arxiv.org/abs/2402.01306>
- [SimPO24] Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward," 2024. <https://arxiv.org/abs/2405.14734>
- [IPO23] Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences," 2023. <https://arxiv.org/abs/2310.12036>
- [DPO23] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023. <https://arxiv.org/abs/2305.18290>
- [DeepSeekMath24] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (introduces GRPO), 2024. <https://arxiv.org/abs/2402.03300>
- [PPO17] Schulman et al., "Proximal Policy Optimization Algorithms," 2017. <https://arxiv.org/abs/1707.06347>
- [TRL] Hugging Face TRL — `DPOTrainer`, `ORPOTrainer`, `KTOTrainer`, `GRPOTrainer`. <https://github.com/huggingface/trl>
- `peft` (LoRA / QLoRA). <https://github.com/huggingface/peft>
- Transformers. <https://github.com/huggingface/transformers>
- Meta-Llama-3.1-8B model cards. <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B> ; <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>
- OWASP Machine Learning Security Top 10 (ML-02 Data Poisoning via preference / binary labels; ML-04 Membership Inference against any of these preference-trained policies; ML-07 Prompt Injection via reward-hacking surfaces in RL-based methods). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — misalignment via corrupted preferences and reward tampering ("Resource Development", "Impact"). <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Govern/Measure/Manage; multi-method surveys and decision trees are Govern/Measure artifacts; per-method failure receipts are Manage). <https://www.nist.gov/itl/ai-risk-management-framework>
