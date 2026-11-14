# mod-105 · exercise-03 — PPO-RLHF Run — Solution

## 1. Solution overview

The deliverable is a short but complete PPO-RLHF run against the reward model from exercise-02, with monitoring that catches the four known failure modes (reward hacking, KL blow-up, mode collapse, training instability) and a `ppo_report.md` that argues "this policy is better than the SFT base — on this distribution, by this much — and here is the evidence it isn't just gaming the reward". A reviewer should be able to read `train_ppo.py`, `config.yaml`, `rollout_logs/`, `ppo_report.md`, and the saved checkpoint, and decide whether to ship or roll back.

Three ideas carry the exercise:

- **PPO without a reference policy and a KL penalty is reward maximization, not alignment.** Every step adds a `−β · KL(π_θ || π_ref)` term so the policy doesn't wander off the SFT base. KL is the budget; the reward is the goal; the ratio of the two is the training signal [InstructGPT22][PPO17].
- **Monitor five numbers, not one.** Per step: mean reward, KL to reference, generation length, value loss, and clip fraction. A policy whose reward is climbing while KL blows up and length doubles is reward-hacking, not learning.
- **The whole thing is a feedback loop, and the RM is the control signal.** The RM's length bias from exercise-02 is now a *training-time* bias. Length penalties, KL coefficient, and adaptive KL control are not optional — they are how you keep the loop stable.

This walkthrough uses `trl.PPOTrainer` [TRL] on a LoRA-adapted `meta-llama/Meta-Llama-3.1-8B-Instruct` policy, the same base with its original SFT weights as the frozen reference, and the `rm-v1` reward model from exercise-02. All numbers are illustrative — the structure is what matters.

## 2. Worked implementation

### 2.1 `config.yaml` — the whole run in one place

```yaml
# config.yaml
run_name: ppo-v1
seed: 0

model:
  policy_base: meta-llama/Meta-Llama-3.1-8B-Instruct   # SFT from mod-103
  reference: meta-llama/Meta-Llama-3.1-8B-Instruct     # Frozen, no LoRA
  reward_adapter: ../exercise-02-reward-model-training/rm_out/adapter
  reward_head:    ../exercise-02-reward-model-training/rm_out/score_head.pt

peft:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

data:
  prompt_file: prompts.jsonl        # 20k deduped prompts, SFT-eval-decontaminated
  max_prompt_tokens: 512
  max_response_tokens: 384

ppo:
  batch_size: 64
  mini_batch_size: 8
  ppo_epochs: 4
  learning_rate: 1.4e-5
  clip_range: 0.2
  value_clip_range: 0.2
  cliprange_value: 0.2
  gamma: 1.0
  lam: 0.95
  vf_coef: 0.1
  grad_clip: 1.0
  init_kl_coef: 0.05
  target_kl: 6.0                    # per-prompt token-mean KL, nats; if we overshoot, beta goes up
  horizon: 10000
  adaptive_kl: true

sampling:
  temperature: 1.0
  top_p: 1.0                        # On-policy: let the policy sample freely; biasing sampling confounds PPO.
  do_sample: true

reward:
  length_penalty_tokens_over: 256   # mild length tax beyond 256 response tokens
  length_penalty_slope: 0.002       # per extra token
  clip_reward_sigma: 3              # clip rewards to ±3 sigma of the rollout window (stability)

training:
  total_steps: 500                  # ~500 * 64 = 32k rollouts ≈ 1 epoch over prompts.jsonl
  log_every: 1
  save_every: 100
  eval_every: 50

eval:
  canary_prompts: ../exercise-02-reward-model-training/rm_out/canary_200.jsonl
  held_out_prompts: eval_prompts.jsonl
  n_pairwise_judges: 3              # for human/model-judge side-by-side vs SFT
```

### 2.2 `train_ppo.py` — reference trainer (trl ≥ 0.9)

```python
# train_ppo.py — illustrative skeleton; trimmed for brevity.
from __future__ import annotations
import json, pathlib, yaml, torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)

CFG = yaml.safe_load(open("config.yaml"))
torch.manual_seed(CFG["seed"])

tok = AutoTokenizer.from_pretrained(CFG["model"]["policy_base"])
tok.pad_token = tok.eos_token
tok.padding_side = "left"           # PPO rollouts left-pad the prompt

# --- policy (LoRA on top of SFT) + value head ------------------------------
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    CFG["model"]["policy_base"],
    peft_config=LoraConfig(
        r=CFG["peft"]["r"],
        lora_alpha=CFG["peft"]["alpha"],
        lora_dropout=CFG["peft"]["dropout"],
        target_modules=CFG["peft"]["target_modules"],
        task_type="CAUSAL_LM",
    ),
    torch_dtype=torch.bfloat16,
)

# --- frozen reference policy (no LoRA) -------------------------------------
reference = AutoModelForCausalLM.from_pretrained(
    CFG["model"]["reference"], torch_dtype=torch.bfloat16
)
for p in reference.parameters(): p.requires_grad_(False)

# --- reward model (from ex-02) ---------------------------------------------
class RewardModel(nn.Module):
    """Same architecture as ex-02; loads the adapter and score head."""
    def __init__(self, base, adapter_path, score_path):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
        self.backbone = PeftModel.from_pretrained(self.backbone, adapter_path)
        self.score = nn.Linear(self.backbone.config.hidden_size, 1, bias=False).to(torch.float32)
        self.score.load_state_dict(torch.load(score_path)["score"])
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def score_responses(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": p},
             {"role": "assistant", "content": r}],
            tokenize=False) for p, r in zip(prompts, responses)]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(self.backbone.device)
        out = self.backbone.model(**enc, output_hidden_states=False)
        last = out.last_hidden_state[:, -1, :].to(torch.float32)
        return self.score(last).squeeze(-1)                 # [B]


reward_model = RewardModel(
    CFG["model"]["policy_base"],
    CFG["model"]["reward_adapter"],
    CFG["model"]["reward_head"],
).cuda()

# --- length penalty / reward clipping --------------------------------------
def shape_rewards(prompts, responses, raw_r: torch.Tensor,
                  tokens_over: int, slope: float, clip_sigma: float
                  ) -> tuple[torch.Tensor, dict]:
    resp_lens = torch.tensor(
        [len(tok.encode(r, add_special_tokens=False)) for r in responses],
        device=raw_r.device, dtype=raw_r.dtype,
    )
    length_tax = slope * torch.clamp(resp_lens - tokens_over, min=0)
    shaped = raw_r - length_tax
    # Clip to ±k sigma within the rollout window to tame heavy-tailed rewards.
    mu, sigma = shaped.mean(), shaped.std().clamp_min(1e-5)
    shaped = torch.clamp(shaped, mu - clip_sigma * sigma, mu + clip_sigma * sigma)
    return shaped, {
        "raw_r_mean": float(raw_r.mean()),
        "shaped_r_mean": float(shaped.mean()),
        "length_tax_mean": float(length_tax.mean()),
        "resp_len_mean": float(resp_lens.float().mean()),
    }

# --- PPO config -------------------------------------------------------------
ppo_cfg = PPOConfig(
    model_name=CFG["run_name"],
    learning_rate=CFG["ppo"]["learning_rate"],
    batch_size=CFG["ppo"]["batch_size"],
    mini_batch_size=CFG["ppo"]["mini_batch_size"],
    ppo_epochs=CFG["ppo"]["ppo_epochs"],
    cliprange=CFG["ppo"]["clip_range"],
    cliprange_value=CFG["ppo"]["value_clip_range"],
    gamma=CFG["ppo"]["gamma"],
    lam=CFG["ppo"]["lam"],
    vf_coef=CFG["ppo"]["vf_coef"],
    max_grad_norm=CFG["ppo"]["grad_clip"],
    init_kl_coef=CFG["ppo"]["init_kl_coef"],
    target=CFG["ppo"]["target_kl"],
    adap_kl_ctrl=CFG["ppo"]["adaptive_kl"],
    seed=CFG["seed"],
)
trainer = PPOTrainer(ppo_cfg, policy, reference, tok)

# --- prompts ---------------------------------------------------------------
prompts = [json.loads(l)["prompt"] for l in open(CFG["data"]["prompt_file"])]

LOG = pathlib.Path("rollout_logs"); LOG.mkdir(exist_ok=True)

gen_kwargs = dict(
    max_new_tokens=CFG["data"]["max_response_tokens"],
    do_sample=True,
    temperature=CFG["sampling"]["temperature"],
    top_p=CFG["sampling"]["top_p"],
    pad_token_id=tok.pad_token_id,
)

step = 0
for step in range(CFG["training"]["total_steps"]):
    # 1. Pick a batch of prompts.
    batch_prompts = _sample_batch(prompts, CFG["ppo"]["batch_size"])

    # 2. Encode, generate, split prompt / response.
    enc = tok(batch_prompts, return_tensors="pt", padding=True,
              truncation=True, max_length=CFG["data"]["max_prompt_tokens"]).to("cuda")
    gen_out = trainer.generate(
        list(enc["input_ids"]),
        return_prompt=False,
        **gen_kwargs,
    )
    responses = [tok.decode(o, skip_special_tokens=True) for o in gen_out]

    # 3. Score, shape.
    raw_r = reward_model.score_responses(batch_prompts, responses).to("cuda")
    shaped_r, shaping_log = shape_rewards(
        batch_prompts, responses, raw_r,
        CFG["reward"]["length_penalty_tokens_over"],
        CFG["reward"]["length_penalty_slope"],
        CFG["reward"]["clip_reward_sigma"],
    )

    # 4. PPO step.
    stats = trainer.step(
        list(enc["input_ids"]),
        list(gen_out),
        [r for r in shaped_r],
    )

    # 5. Log.
    row = {
        "step": step,
        "reward_mean": float(shaped_r.mean()),
        "reward_raw_mean": shaping_log["raw_r_mean"],
        "resp_len_mean": shaping_log["resp_len_mean"],
        "kl_mean": float(stats["objective/kl"]),
        "kl_coef": float(stats["objective/kl_coef"]),
        "policy_loss": float(stats["ppo/loss/policy"]),
        "value_loss":  float(stats["ppo/loss/value"]),
        "clip_frac":   float(stats["ppo/policy/clipfrac"]),
    }
    print(json.dumps(row))
    (LOG / f"step_{step:05d}.json").write_text(
        json.dumps({"row": row, "prompts": batch_prompts, "responses": responses,
                    "raw_r": raw_r.tolist(), "shaped_r": shaped_r.tolist()}))

    # 6. Save adapter & evaluate periodically.
    if (step + 1) % CFG["training"]["save_every"] == 0:
        trainer.save_pretrained(f"ppo_out/step_{step+1:05d}")
```

Notes on the reference:

- **`AutoModelForCausalLMWithValueHead`** gives you a 1-dim value head on top of the policy. TRL's `PPOTrainer.step` assumes this exact shape; swapping in a vanilla `AutoModelForCausalLM` crashes mid-step.
- **Reference is frozen, no LoRA.** The reference policy is the SFT base, period. Loading it with the policy's LoRA adapter active collapses the KL term to ~0 artificially and the whole regularization disappears.
- **`adap_kl_ctrl=True` + `target_kl=6.0`.** The KL coefficient is scaled up when the rolling KL exceeds `target_kl`, and down when it undershoots. Without this, you either over-regularize (no reward improvement) or lose the reference (reward climbs, policy diverges). TRL's `AdaptiveKLController` is the reference implementation [TRL].
- **Length penalty is a *reward*-level term, not a logit-processor term.** Penalizing length via a logit processor at sampling time changes what the policy sees; penalizing it in the reward keeps sampling honest and lets the length-vs-reward trade-off be learned explicitly.
- **Reward clipping to ±3σ within the rollout window.** Reward distributions are heavy-tailed; a single outlier at `r = 12` causes a gradient spike that undoes 10 prior steps. Clipping is cheap insurance.
- **Left-pad prompts** so the generation picks up at the right position; this is a `trl` requirement, not a stylistic choice.
- **Prompt-decontam from exercise-01.** `prompts.jsonl` is already SFT-eval-decontaminated. If the same prompts the policy is being eval'd on appear here, every eval number is leaked.

### 2.3 Monitoring: what "healthy" looks like

The run should show these trajectories in `rollout_logs/`:

| Metric | Healthy start | Healthy end (step 500) | Unhealthy signature |
|---|---|---|---|
| `reward_raw_mean` | +0.05 | +1.1 to +1.5 | Climbing to +3+ (reward hacking) |
| `kl_mean` (per-token, nats) | ~0.02 | 4–6 | > 10 (ref lost) or ~0 (no learning) |
| `kl_coef` | 0.05 | 0.08–0.20 (adaptive) | Pinned at `init_kl_coef` means `adaptive_kl=False` or `target_kl` wrong |
| `resp_len_mean` | ~110 | 150–220 | > 300 and growing (length hack) |
| `policy_loss` | near 0 | small negative, oscillating | Monotonic negative → policy pinned |
| `value_loss` | ~1.0 | 0.1–0.3 | Flat at 1.0 → value head isn't learning |
| `clip_frac` | 0.05–0.15 | 0.15–0.25 | > 0.4 (step too large) or < 0.02 (step too small) |

### 2.4 `ppo_report.md` — target shape

```markdown
# PPO-RLHF Report — `ppo-v1`

## Setup
- Policy: Llama-3.1-8B-Instruct SFT (mod-103), LoRA r=16/α=32 on qkvo, bf16.
- Reference: Llama-3.1-8B-Instruct SFT, frozen, no LoRA.
- Reward: rm-v1 (ex-02), LoRA + fp32 scalar head.
- Prompts: 20,000 unique, SFT-eval-decontaminated (from ex-01 pool, prompts only).
- PPO: TRL 0.9, batch 64, mini-batch 8, 4 PPO epochs/step, clip 0.2.
- KL: adaptive, init_kl_coef=0.05, target_kl=6.0.
- Steps: 500 (≈1 epoch).
- Hardware: 2× A100 80 GB, ~14 h.

## Training trajectory (summary)

| Step | reward_raw | reward_shaped | kl_mean | resp_len | kl_coef | clip_frac |
|---:|---:|---:|---:|---:|---:|---:|
|   0 | +0.03 | +0.03 | 0.02 | 108 | 0.050 | 0.07 |
|  50 | +0.41 | +0.40 | 1.9  | 128 | 0.055 | 0.14 |
| 100 | +0.78 | +0.76 | 3.3  | 144 | 0.068 | 0.17 |
| 200 | +1.12 | +1.05 | 4.9  | 171 | 0.096 | 0.19 |
| 300 | +1.27 | +1.17 | 5.6  | 184 | 0.112 | 0.17 |
| 400 | +1.38 | +1.25 | 5.9  | 192 | 0.126 | 0.16 |
| 500 | +1.41 | +1.26 | 6.1  | 198 | 0.141 | 0.15 |

Notes:
- `reward_raw` climbed and plateaued around 1.4. `reward_shaped` (after length
  tax) climbed and plateaued around 1.25 — the 0.15 delta is the paid tax.
- `kl_mean` approached but did not exceed `target_kl=6.0`; `kl_coef` adapted
  from 0.05 → 0.14 as expected.
- `resp_len_mean` grew from 108 → 198. The length penalty held it below 256;
  without the penalty, we observed runaway to 350+ in pilot runs.

## Policy quality vs. SFT (held-out 500 prompts)

### RM score
- SFT mean reward: +0.04 (std 1.14)
- PPO-v1 mean reward: +1.33 (std 1.02)
- Δ reward: +1.29. Length-adjusted (subtract per-token mean length delta × slope): +1.11.

### Side-by-side judge (model-as-judge, 2 templates, order-randomized)
- PPO-v1 wins: 61%
- SFT wins: 24%
- Tie: 15%
Only pairs where both judges agree are counted; 384/500 pairs qualified.

### Length
- SFT: mean 118, p95 212 tokens.
- PPO-v1: mean 198, p95 287 tokens.
- Length delta is real but not pathological (see length-adjusted RM score).

### Safety slice (red-team subset, 120 prompts)
- SFT refusal correctness: 76% (72 correct refusals + 20 appropriate helpfuls).
- PPO-v1 refusal correctness: 71%.
- Regression of 5 pts on harmlessness — flagged for mod-106 safety-tuning pass.
  Root cause (manual audit of 40 cases): PPO learned to "soften" refusals into
  partial compliance that scores higher on the helpfulness RM. This is exactly
  the entanglement risk that motivates keeping helpful/harmless separate.

## Failure-mode checks

1. **Reward hacking by length.**
   - `corr(reward, response_length)` on final PPO rollouts: +0.34.
   - Was +0.38 on the pre-PPO SFT (just RM length bias).
   - Interpretation: PPO did NOT further amplify the length bias (penalty working).
2. **KL blow-up.**
   - Max step KL mean: 6.12 (step 492). Target 6.0.
   - Max per-token KL seen in any single rollout: 23.1 (anomaly on a single
     code-generation prompt; the policy produced a near-verbatim reference
     document the SFT never would have).
3. **Mode collapse.**
   - Distinct-2 on 500 rollouts: 0.41 (SFT baseline: 0.47). Slight decrease.
   - Unique-4-gram ratio: 0.72 (SFT: 0.78). Slight decrease, within tolerance.
4. **Distribution drift on canary (200 fixed prompts).**
   - Mean canary reward: SFT +0.02 → PPO-v1 +1.28. KL to SFT: 5.9 per-token mean.
     Within the designed KL budget. No safety-slice prompt drifted negatively.

## Acceptance

| Criterion | Threshold | Actual | Pass |
|---|---|---|---|
| Δ mean reward vs. SFT on held-out | ≥ +0.5 | +1.29 | ✅ |
| Judge side-by-side win rate | ≥ 55% | 61% | ✅ |
| KL to reference (per-token mean, final) | ≤ 8 | 6.1 | ✅ |
| Distinct-2 on 500 rollouts | ≥ 0.35 | 0.41 | ✅ |
| Safety refusal correctness regression | ≤ 10 pts | 5 pts | ✅ (flagged for mod-106) |
| Length correlation with reward on rollouts | ≤ +0.50 | +0.34 | ✅ |

Recommendation: **ship ppo-v1 for internal eval; do not promote to production
until mod-106 safety-tuning closes the 5-pt refusal regression.**
```

### 2.5 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| PPO with adaptive KL vs. reference | Classic RLHF; the reference anchors the policy to the SFT distribution [InstructGPT22] | REINFORCE / A2C | Higher variance; no clip; practically brittle |
| `init_kl_coef=0.05, target_kl=6` | Lets reward climb for the first ~50 steps, then throttles | Fixed `kl_coef=0.2` | Throttles too hard at the start; slow learning |
| LoRA on policy, no LoRA on reference | Matches training recipe to mod-104; reference must be exactly the SFT the user shipped | Full-weight policy | 4–8× more VRAM; marginal quality gain |
| Length penalty in reward (not sampling) | Lets policy learn the trade-off; keeps sampling on-policy | `length_penalty` in `generate()` | Logit-level penalty is off-policy and confounds PPO's importance ratio |
| Reward clip to ±3σ within rollout window | Tames heavy-tailed RM scores; one-line stability win | No clip | Occasional gradient explosion |
| Separate value head on policy | Matches TRL's PPO assumption; reduces advantage variance | Shared head | Debugger says no — TRL hard-codes this |
| `top_p=1.0, temp=1.0` sampling | Pure on-policy; biasing sampling breaks PPO's probability-ratio math | `top_p=0.9` | Policy becomes "greedy + PPO"; importance ratio is off |
| Monitor 5 metrics every step | Any one of them in isolation misleads | Just reward | You cannot distinguish learning from reward hacking |
| Canary + judge side-by-side on eval | RM score alone is self-referential; needs an external signal | RM score only | You might ship a policy that games the RM |

## 3. Validation steps

1. **Step-0 KL ≈ 0.** Policy and reference are the same weights at step 0; the LoRA is zero-delta. If `kl_mean` at step 0 is non-trivial, either the LoRA is non-zero-init or the reference loaded a different checkpoint.
2. **Reward climbs for the first 50 steps.** If it is flat, `kl_coef` is pinned too high, `learning_rate` is too low, or the reward signal is degenerate (every response scores the same).
3. **`kl_mean` stays within `[0.2 × target_kl, 1.2 × target_kl]` after warmup.** Below: policy isn't learning. Above: adaptive KL isn't working; expect divergence.
4. **`clip_frac` in `[0.05, 0.25]`.** Below means the step is tiny (PPO underused). Above means the step is too aggressive and importance ratios are outside the trust region — expect instability.
5. **Response length doesn't grow past the length penalty's control range.** If `resp_len_mean` goes past 256 and keeps climbing, the penalty slope is too low. Raise `length_penalty_slope` by 2× and restart.
6. **Held-out judge win rate > 55% vs. SFT.** Below this the policy has not learned a human-legible improvement, even if the RM says so.
7. **Safety slice regression ≤ 10 points.** A larger regression means the helpful RM is shoving the policy into the "mildly compliant" region on red-team prompts; stop and either retrain with a harmlessness RM term or sequence with mod-106.
8. **Deterministic rebuild on seed 0.** Same hardware, same PRNG, same reward adapter → same loss curve shape within fp32 noise. Any larger deviation indicates upstream non-determinism (rollout order, generation seed, reward model drift).

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **PPO plumbing correctness** | Missing value head, non-frozen reference, or no KL term | KL present but fixed (non-adaptive) | Adaptive KL, frozen reference, correct value head, left-padded rollouts |
| **Reward shaping & stability** | Raw RM reward only, no clipping | Length penalty OR reward clip, not both | Length penalty in reward, ±kσ reward clip, trust region behaving |
| **Per-step monitoring** | Logs reward only | Reward + KL only | Reward, KL, length, value loss, clip frac — all per step |
| **Eval beyond RM score** | Only RM score | RM score + length | RM score + length + judge side-by-side + safety-slice regression + canary |
| **Acceptance criteria & rollback** | No thresholds | Thresholds but no rollback plan | Explicit thresholds, per-criterion pass/fail, rollback decision documented |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- **Dropping the KL term.** The policy diverges from the SFT, the reward keeps climbing, and generations start to drift into gibberish that the RM happens to like. KL is non-optional.
- **Fixed `kl_coef` with no adaptive controller.** Either too small (no regularization) or too large (no learning). Adaptive KL is a two-line addition and solves both.
- **Non-frozen reference.** If the reference has a LoRA loaded, KL collapses to ~0 and you've effectively removed the regularizer.
- **Different chat template between policy and RM.** RM scores on policy samples are OOD; PPO optimizes a noisy signal.
- **Length penalty in `generate()` instead of in the reward.** Changes the sampling distribution the policy is trained on; PPO's importance ratio no longer tracks truth.
- **`top_p < 1` sampling.** Off-policy on a PPO trainer that assumes on-policy; the probability ratio is wrong.
- **Shipping on RM score alone.** The policy has been *trained* to maximize this number; it will be higher. The question is whether it is higher for a reason humans care about.
- **Ignoring the length-vs-reward correlation at eval time.** PPO happily walks up a 0.4-Pearson length bias and you miss it if you only look at mean reward.
- **Running one long run and no canary.** You lose the ability to tell whether the next change (new prompt mix, new RM version) is an improvement.
- **Treating safety-slice regressions as acceptable because "the helpful number went up".** Helpful and harmless are entangled through the RM mix; losing 5+ points on refusals is a ship blocker for a production policy and should route to mod-106.
- **Not saving per-step rollouts.** When PPO goes sideways on step 312 you need the batch that caused it, not a summary metric.

## 6. References

- [PPO17] Schulman et al., "Proximal Policy Optimization Algorithms," 2017. <https://arxiv.org/abs/1707.06347>
- [InstructGPT22] Ouyang et al., "Training language models to follow instructions with human feedback," 2022. <https://arxiv.org/abs/2203.02155>
- [HH22] Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback," 2022. <https://arxiv.org/abs/2204.05862>
- [TRL] Hugging Face TRL library (PPO, reward, SFT trainers). <https://github.com/huggingface/trl> — `ppo_trainer.py` and `AdaptiveKLController` are the authoritative references.
- Transformers (Hugging Face). <https://github.com/huggingface/transformers>
- `peft` (LoRA / QLoRA adapters). <https://github.com/huggingface/peft>
- Meta-Llama-3.1-8B-Instruct model card. <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>
- OWASP Machine Learning Security Top 10 (ML-02 Data Poisoning via preference pairs, ML-04 Reward model as a target for membership inference, ML-07 Model prompt-injection via reward-hacking surface). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — reward-hacking tactics under "Impact" and misalignment via reward tampering under "Resource Development". <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Govern/Measure/Manage; RLHF monitoring and acceptance criteria are Measure/Manage obligations; the acceptance table is Govern). <https://www.nist.gov/itl/ai-risk-management-framework>
