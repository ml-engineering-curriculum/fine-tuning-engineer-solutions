# mod-105 · exercise-02 — Reward Model Training — Solution

## 1. Solution overview

The deliverable is a Bradley–Terry reward model trained on the preference pairs from exercise-01, evaluated on a held-out pair test set, and audited for the one failure mode that matters most for RLHF: **reward hacking by length**. A reviewer should be able to read `train_rm.py`, `eval_rm.py`, `rm_report.md`, and the saved checkpoint, and sign off that the RM is calibrated well enough to feed PPO (exercise-03) or to be reused as a judge for DPO (exercise-04).

Three ideas carry the exercise:

- **A reward model is a classifier, not a regressor.** It maps a `(prompt, response)` into a scalar score, trained with a pairwise log-sigmoid loss so that `r(chosen) > r(rejected)` under the Bradley–Terry model [BT52], [InstructGPT22]. Treating it as a regression on human scores is the classic beginner mistake — human raters aren't calibrated, pairs are.
- **Validation pair accuracy is a *necessary* number, not a sufficient one.** A 70%-pair-accuracy RM that hates long responses will steer PPO into terse answers regardless of helpfulness. Every RM report must include: pair accuracy, calibration curve, length correlation, and score drift across a safety / style slice.
- **The architecture is boring; the loss and the eval are not.** A frozen base + a 1-token score head + pairwise log-sigmoid is the whole model. 90% of what's hard lives in data handling (same tokenization as the policy will use, same chat template, truncation that doesn't chop the chosen in half), evaluation (slice-wise accuracy), and bias diagnostics.

This walkthrough trains on the `prefs-v1` pairs from exercise-01, with `meta-llama/Meta-Llama-3.1-8B-Instruct` as the base (LoRA-adapted; a full-weight RM is strictly optional) on a single 80 GB A100. The structure generalizes unchanged to any decoder-only base with a `nn.Linear` lm_head.

## 2. Worked implementation

### 2.1 `train_rm.py` — the reference trainer

```python
# train_rm.py — illustrative skeleton; trimmed for brevity, not correctness.
from __future__ import annotations
import json, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

BASE = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PREFS = "out/pairs_train.jsonl"
VAL = "out/pairs_val.jsonl"
OUT = pathlib.Path("rm_out"); OUT.mkdir(exist_ok=True)
MAX_LEN = 2048
SEED = 0

torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token      # RM pads on left so the last-token head is the actual last content token.
tok.padding_side = "left"


def render(prompt: str, response: str) -> list[int]:
    # Use the SAME template the policy will use at generation time.
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # Truncate from the LEFT so we never chop the response. If the prompt is
    # too long, the model sees a truncated prompt; that's fine — the response
    # is what carries the reward signal.
    if len(ids) > MAX_LEN:
        ids = ids[-MAX_LEN:]
    return ids


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines()]

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        c = render(r["prompt"], r["chosen"])
        j = render(r["prompt"], r["rejected"])
        # Trust weight: pass through as a per-pair scalar in [0.5, 1.0].
        w = _trust_weight(r)
        return {
            "chosen_ids": torch.tensor(c, dtype=torch.long),
            "rejected_ids": torch.tensor(j, dtype=torch.long),
            "weight": torch.tensor(w, dtype=torch.float),
            "dim": r["preference_dimension"],
        }


def _trust_weight(row: dict) -> float:
    src = row["source"]
    base = {
        "hh-rlhf:helpful-base": 0.7,
        "hh-rlhf:harmless-base": 0.8,
        "ultrafeedback-binarized": 0.6,
        "oasst2": 0.9,
        "synthetic": 0.5,
    }.get(src.split(":")[0] if "-rlhf" not in src else src, 0.5)
    # Up-weight pairs with a large rank/score gap.
    agree = row.get("agreement", {})
    if agree.get("kind") == "pointwise_binarized":
        base += 0.1 * min(1.0, (float(agree["value"]) - 1.0) / 4.0)
    if agree.get("kind") == "ranking":
        delta = row["rejected_rank_meta"]["rank"] - row["chosen_rank_meta"]["rank"]
        base += 0.05 * min(1.0, max(0.0, delta - 2.0))
    return max(0.5, min(1.0, base))


def pad_collate(batch):
    def pad(ids_list):
        max_l = max(x.size(0) for x in ids_list)
        pad_id = tok.pad_token_id
        out = torch.full((len(ids_list), max_l), pad_id, dtype=torch.long)
        mask = torch.zeros((len(ids_list), max_l), dtype=torch.long)
        for i, x in enumerate(ids_list):
            # left pad
            out[i, max_l - x.size(0):] = x
            mask[i, max_l - x.size(0):] = 1
        return out, mask

    c_ids, c_mask = pad([b["chosen_ids"] for b in batch])
    j_ids, j_mask = pad([b["rejected_ids"] for b in batch])
    weights = torch.stack([b["weight"] for b in batch])
    dims = [b["dim"] for b in batch]
    return c_ids, c_mask, j_ids, j_mask, weights, dims


class RewardModel(nn.Module):
    """Frozen base + a single scalar head over the last non-pad position."""
    def __init__(self, base_name: str):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16
        )
        # LoRA adapts attention projections, matches PEFT conventions from mod-104.
        self.backbone = get_peft_model(self.backbone, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ))
        hid = self.backbone.config.hidden_size
        self.score = nn.Linear(hid, 1, bias=False).to(torch.float32)
        nn.init.zeros_(self.score.weight)                    # start at r = 0 everywhere

    def forward(self, input_ids, attention_mask):
        out = self.backbone.base_model.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hs = out.last_hidden_state                           # [B, T, H], bf16
        # Left-pad → the last token position IS the content's last token.
        last = hs[:, -1, :].to(torch.float32)                # [B, H]
        return self.score(last).squeeze(-1)                  # [B]


def pairwise_loss(r_c: torch.Tensor, r_j: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Bradley–Terry: log sigmoid(r_c - r_j). Weighted mean.
    return -(w * F.logsigmoid(r_c - r_j)).mean()


def main():
    train_ds = PairDataset(PREFS)
    val_ds = PairDataset(VAL)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              collate_fn=pad_collate, num_workers=2,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                            collate_fn=pad_collate, num_workers=2)

    rm = RewardModel(BASE).to("cuda")
    opt = torch.optim.AdamW(
        [p for p in rm.parameters() if p.requires_grad],
        lr=5e-6, weight_decay=0.0,
    )
    sched = get_linear_schedule_with_warmup(opt, 50, len(train_loader) * 2)  # 2 epochs

    step = 0
    for epoch in range(2):
        rm.train()
        for c_ids, c_mask, j_ids, j_mask, w, dims in train_loader:
            c_ids, c_mask = c_ids.cuda(), c_mask.cuda()
            j_ids, j_mask = j_ids.cuda(), j_mask.cuda()
            w = w.cuda()
            r_c = rm(c_ids, c_mask)
            r_j = rm(j_ids, j_mask)
            loss = pairwise_loss(r_c, r_j, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in rm.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step()
            if step % 50 == 0:
                print(f"step={step} loss={loss.item():.4f} "
                      f"acc_batch={(r_c > r_j).float().mean().item():.3f}")
            step += 1

        val = _eval(rm, val_loader)
        print(f"epoch={epoch} val_acc={val['acc']:.3f} val_loss={val['loss']:.4f}")

    rm.backbone.save_pretrained(OUT / "adapter")
    torch.save({"score": rm.score.state_dict()}, OUT / "score_head.pt")


@torch.no_grad()
def _eval(rm: nn.Module, loader) -> dict:
    rm.eval()
    losses, correct, n = [], 0, 0
    for c_ids, c_mask, j_ids, j_mask, w, _ in loader:
        c_ids, c_mask = c_ids.cuda(), c_mask.cuda()
        j_ids, j_mask = j_ids.cuda(), j_mask.cuda()
        w = w.cuda()
        r_c = rm(c_ids, c_mask); r_j = rm(j_ids, j_mask)
        losses.append(pairwise_loss(r_c, r_j, w).item())
        correct += (r_c > r_j).sum().item()
        n += r_c.size(0)
    return {"acc": correct / n, "loss": sum(losses) / len(losses)}


if __name__ == "__main__":
    main()
```

Notes on the reference:

- **Left-pad + last-position head** is the convention that matches most open RM implementations (`trl.AutoModelForSequenceClassification`, `trl.reward_trainer.RewardTrainer`) and lets us read the score in one line. Right-padding with a mask-aware pooling works, but the last-position trick removes an off-by-one in attention mask bookkeeping.
- **Score head in fp32, LoRA in bf16.** Keeping the 1-dim head in fp32 prevents log-sigmoid from saturating in bf16 when `|r_c − r_j|` grows, which it will after an epoch. This is the single most common "loss went to zero but accuracy is still 55%" bug in practice.
- **`nn.init.zeros_(self.score.weight)`** means `r = 0` for every `(prompt, response)` at step 0. The log-sigmoid loss is `log 2` everywhere, and accuracy is `~0.5`. If you don't see those numbers at step 0, something else is initialized non-trivially.
- **Trust weighting** is a cheap way to use per-source reliability from the data card. If you skip it, UFB (noisier) and OASST (cleaner) have equal pull on the RM. The weight stays in `[0.5, 1.0]`; going below 0.5 effectively discards a source and is better handled by removing the source.
- **Same template as the policy will use at generation time.** If PPO renders `[INST] ... [/INST]` and the RM was trained on `<|user|>...<|assistant|>`, the RM's scores on policy samples are out-of-distribution and the reward signal is unreliable. This is the single largest "RM worked on paper, PPO collapsed in reality" failure.

### 2.2 `eval_rm.py` — the acceptance eval

Pair accuracy alone is misleading. The acceptance protocol is pair accuracy **plus** four diagnostics.

```python
# eval_rm.py — produces rm_report.md numbers.
import json, numpy as np, torch
from scipy.stats import pearsonr, spearmanr

def accuracy_by_slice(records):
    # records: [{dim, source, r_c, r_j, len_c, len_j, agree_margin}, ...]
    def _acc(rs): return sum(1 for r in rs if r["r_c"] > r["r_j"]) / max(1, len(rs))
    overall = _acc(records)
    by_dim    = {d: _acc([r for r in records if r["dim"] == d])
                 for d in {r["dim"] for r in records}}
    by_source = {s: _acc([r for r in records if r["source"] == s])
                 for s in {r["source"] for r in records}}
    return {"overall": overall, "by_dim": by_dim, "by_source": by_source}


def length_bias(records):
    # Does the RM prefer longer responses regardless of label?
    r_delta = np.array([r["r_c"] - r["r_j"] for r in records])
    l_delta = np.array([r["len_c"] - r["len_j"] for r in records])
    pearson, _ = pearsonr(r_delta, l_delta)
    spearman, _ = spearmanr(r_delta, l_delta)
    # If length is uncorrelated with the TRUE label (approx 0 in train), a
    # Pearson > 0.5 between reward-delta and length-delta flags learned bias.
    return {"pearson_r_delta_vs_len_delta": float(pearson),
            "spearman_r_delta_vs_len_delta": float(spearman)}


def calibration(records, n_bins=10):
    # Expected vs. empirical: group pairs by |r_c - r_j|, compute win rate.
    gaps = np.abs([r["r_c"] - r["r_j"] for r in records])
    wins = np.array([r["r_c"] > r["r_j"] for r in records], dtype=float)
    # Bin by gap decile.
    order = np.argsort(gaps)
    gaps, wins = gaps[order], wins[order]
    bins = np.array_split(np.arange(len(gaps)), n_bins)
    rows = []
    for b in bins:
        if len(b) == 0: continue
        gap_mid = gaps[b].mean()
        pred = 1 / (1 + np.exp(-gap_mid))         # Bradley–Terry predicted prob
        emp = wins[b].mean()
        rows.append({"gap_mean": float(gap_mid),
                     "predicted_win_rate": float(pred),
                     "empirical_win_rate": float(emp),
                     "n": int(len(b))})
    return rows


def judge_distribution_drift(rm_scores_on_seed_prompts, policy_type):
    # Mean / std of r on a fixed 200-prompt "canary" seed set across
    # candidate policies. Drift > 0.5 sigma between RM versions = refuse to ship.
    return {"mean": float(np.mean(rm_scores_on_seed_prompts)),
            "std": float(np.std(rm_scores_on_seed_prompts)),
            "policy": policy_type}
```

### 2.3 `rm_report.md` — target shape

```markdown
# Reward Model Report — `rm-v1`

## Setup
- Base: meta-llama/Meta-Llama-3.1-8B-Instruct (bf16), LoRA r=16/α=32 on qkvo.
- Head: 1-dim linear, fp32, zero-initialized.
- Loss: Bradley–Terry log-sigmoid, weighted by per-source trust (0.5–1.0).
- Train / Val / Test: prefs-v1 (4,481 / 527 / 264 pairs).
- Max seq length: 2,048 (left-padded, left-truncated).
- Optimizer: AdamW(lr=5e-6), 50 warmup, 2 epochs, grad clip 1.0.
- Seed: 0. Hardware: 1× A100 80 GB. Wall clock: ~2.3 h.

## Headline accuracy

| Split | Pairs | Pair acc | Weighted loss |
|---|---:|---:|---:|
| train (last epoch) | 4,481 | 0.812 | 0.408 |
| val | 527 | 0.727 | 0.542 |
| test | 264 | 0.714 | 0.555 |

## Slice-wise accuracy (test)

| Slice | n | Pair acc |
|---|---:|---:|
| dim=helpful | 203 | 0.724 |
| dim=harmless | 61 | 0.689 |
| src=hh-rlhf:helpful-base | 94 | 0.702 |
| src=hh-rlhf:harmless-base | 61 | 0.689 |
| src=ultrafeedback-binarized | 76 | 0.737 |
| src=oasst2 | 25 | 0.760 |
| src=synthetic | 8 | 0.750 |

Interpretation: helpfulness is ~0.72, harmlessness is ~0.69. OASST is the
easiest source (highest rank-delta threshold); UFB is close behind because
it is pointwise-binarized, which the RM finds easy to re-rank. hh-rlhf is
hardest — long dialogs with subtle preference drivers.

## Length bias

| Metric | Value |
|---|---:|
| Pearson(r_c − r_j, len_c − len_j), test | +0.38 |
| Spearman(r_c − r_j, len_c − len_j), test | +0.41 |
| Pearson on dim=helpful only | +0.44 |
| Pearson on dim=harmless only | +0.09 |

Interpretation: the helpfulness head has a moderate length preference
(cf. [UFB23] discussion of GPT-4-judge length bias); harmlessness is flat.
PPO (exercise-03) MUST use a length-penalty term and a KL regularizer, or
the policy will trivially grow longer to climb the reward.

## Calibration (Bradley–Terry)

| |r_c − r_j| bin | Predicted win | Empirical win | n |
|---:|---:|---:|---:|
| 0.1 | 0.525 | 0.531 | 26 |
| 0.4 | 0.598 | 0.615 | 27 |
| 0.9 | 0.711 | 0.741 | 27 |
| 1.5 | 0.818 | 0.815 | 27 |
| 2.3 | 0.909 | 0.889 | 27 |
| 3.8 | 0.978 | 0.963 | 27 |

Interpretation: calibration is good up to |Δr| ≈ 2; the model becomes
slightly over-confident past |Δr| ≈ 3. This is a Bradley–Terry-fit
feature, not a bug. PPO should treat anything with |Δr| > 3 as "roughly
equally high reward" rather than as a hard ranking signal.

## Canary distribution (200 fixed seed prompts)

| Candidate policy | Mean reward | Std |
|---|---:|---:|
| Base SFT (mod-103 artifact) | +0.02 | 1.18 |
| Base SFT + length prompt ("be concise") | −0.11 | 1.25 |
| Base SFT + length prompt ("be thorough") | +0.47 | 1.31 |

Interpretation: a trivial length prompt moves reward by ~0.5 absolute. Any
PPO run that shows reward climbing by less than this per epoch is probably
just learning length, not helpfulness.

## Failure analysis (hand-audited 40 worst predictions on val)

- 18/40: `rejected` was actually fine; `chosen` won on a stylistic tie.
  Source: UFB low-margin pairs. Fix: raise the margin floor to 1.5 in
  exercise-01 v2.
- 9/40: both responses are safety refusals of a benign prompt. Reward
  signal is uninformative. Fix: flag at ingest.
- 7/40: formatting-only differences (bulleted vs. prose). Not a real
  helpfulness signal; RM is fine to be noisy here.
- 6/40: red-team jailbreak attempts where the "chosen" response is a
  partial refusal; genuine edge case for the harmlessness RM.

## Acceptance

- Pair accuracy on test: 0.714 ≥ threshold 0.65.
- Length bias (Pearson on helpful): +0.44 ≤ ceiling 0.60.
- Calibration gap at top bin: |0.963 − 0.978| = 0.015 ≤ 0.05.
- Canary drift: within ±0.5 of prior rm-v0 on 190/200 prompts.

Recommendation: **accept for PPO/DPO use with length penalty required**.
```

### 2.4 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Bradley–Terry pairwise log-sigmoid | Matches [InstructGPT22] and all open RMs; consumes exactly the pairs we built | MSE on pointwise scores | Human rater scores aren't calibrated; MSE underperforms by 3–7 pts pair acc |
| Last-position scalar head on left-padded input | Avoids attention-mask pooling bugs; one matmul | Mean-pool over response tokens | Mean pool is less sensitive to the last conclusion sentence; usually worse on long responses |
| Head init = zeros, in fp32 | Deterministic step-0 state (r=0, loss=log 2); avoids bf16 log-sigmoid saturation later | Small random head | Non-zero step-0 loss; harder to debug numerical issues |
| LoRA-only adapts the backbone | Memory budget fits on a single 80 GB card; matches the policy-side training regime from mod-104 | Full-weight RM | ~0.5–1 pt better pair acc at ~4× VRAM and training time |
| Trust weights per source | Preference-pair data card already encodes reliability; discarding it throws away signal | Uniform weight | ~1–2 pts worse val acc in practice (data-dependent) |
| Separate rewards for helpful vs. harmless (via two RMs or a multi-head) | Different trade-offs during RL; one scalar entangles them | Single mixed RM | Easy to ship but PPO oscillates between terseness and politeness |
| Left truncation at 2,048 tokens | Preserves the response (the signal); drops prompt prefix | Right truncation | Right truncation sometimes chops the chosen answer's conclusion — catastrophic |
| Report length bias as part of acceptance | It is the #1 hacking vector for PPO downstream | Only report pair acc | Reports look cleaner but PPO blows up |

## 3. Validation steps

1. **Step-0 sanity.** On construction, `rm(x)` returns all-zero scores. Pair loss at step 0 is `log 2 ≈ 0.693`. Batch accuracy is `~0.5`. Failing any of these means the head is not zero-init, or the frozen weights have drift.
2. **Pair accuracy on held-out test ≥ 0.65.** Below this the RM is not reliably better than coin-flip-weighted-by-length. Below 0.60 abort — go back to exercise-01 and tighten margins / fix labels.
3. **Length bias Pearson ≤ 0.6.** Above this, PPO (exercise-03) will reward-hack by length within the first epoch. If Pearson > 0.6 and the RM cannot be retrained, mitigation is a length penalty term plus a tighter KL (see exercise-03's `kl_coef` and `len_penalty`).
4. **Calibration gap ≤ 0.05 in the top |Δr| bin.** If the empirical win rate lags the predicted win rate by more than 5 points, the RM is overconfident on the easy pairs; PPO over-commits to its current best response.
5. **Canary drift against prior RM version ≤ 0.5 reward on 95% of 200 fixed prompts.** New RM shouldn't disagree wildly with the prior one on a random held-out set; if it does, PPO's starting dynamics change dramatically and any A/B comparison between policy versions is confounded.
6. **No eval-set leakage.** Decontam check from exercise-01 covered the SFT eval; re-run against the preference test set to confirm there is no prompt the RM was trained on that also appears in the policy's eval harness.
7. **Deterministic rebuild.** Training on seed 0, same hardware, same hyperparameters must reproduce val accuracy to the second decimal.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Loss / head correctness** | MSE regression on scalar scores, or sigmoid-of-concatenated-logits | Bradley–Terry loss but non-zero-init head or bf16 head | B–T log-sigmoid, fp32 scalar head, zero-init, passes step-0 sanity |
| **Data pipeline consistency with policy** | Different tokenizer, template, or truncation side than the eventual policy | Same tokenizer and template but right-truncation (clips response) | Same tokenizer, chat template, and left-truncation; matches policy's generation stack |
| **Trust / confidence weighting** | Uniform weights, no use of exercise-01 metadata | Flat per-source weight only | Per-source trust × margin/Δrank up-weight on each pair |
| **Eval beyond pair accuracy** | Only pair accuracy reported | Pair accuracy + slice breakdown | Accuracy + slice + length bias + calibration + canary |
| **Report & acceptance criteria** | Numbers but no acceptance thresholds | Thresholds but no justification | Thresholds with per-metric justification + sign-off for PPO/DPO |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- **Treating the RM as a regressor.** Fitting MSE to raw rater scores ignores the fact that pairs are the actual signal; pair accuracy ends up 5–10 points lower than Bradley–Terry.
- **Right-truncation.** Clipping the end of the response removes the thing the RM should reward. Pair accuracy drops silently on long-response slices.
- **Using a different chat template than the policy.** The RM's scores on policy samples become out-of-distribution; PPO loss looks fine, generations drift into mode collapse.
- **Non-zero head init.** Step-0 loss is no longer `log 2`; you can't tell whether the drop in epoch 1 is learning or regression-to-mean.
- **bf16 everywhere.** Log-sigmoid saturates on large positive `r_c − r_j`; gradient becomes ~0 and training stalls. Score head in fp32 fixes it in one line.
- **Pair-accuracy-only reporting.** 70% pair accuracy with a 0.7 length Pearson is a worse RM for RLHF than 65% pair accuracy with a 0.2 length Pearson. The numbers don't tell the story without the bias slices.
- **Mixing dimensions into one head.** A single RM trained on both helpfulness and harmlessness learns an arbitrary weighted average; PPO oscillates between "be terse / safe" and "be thorough / helpful".
- **Not stratifying val/test by source.** A single dominant source can make val look great and test collapse when the pool mix changes in exercise-01 v2.
- **Skipping the canary set.** When you retrain the RM, PPO behavior changes. Without a 200-prompt canary distribution you can't tell if the next policy run is better because the policy learned or because the RM moved.
- **Reporting `loss.item()` as if it were small.** A 0.4 training loss on log-sigmoid with mean |Δr| = 2 is a model with ~70% pair acc, not "very low loss". Always report accuracy, not just loss.

## 6. References

- [BT52] Bradley, Terry, "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons," 1952. <https://doi.org/10.2307/2334029>
- [InstructGPT22] Ouyang et al., "Training language models to follow instructions with human feedback," 2022. <https://arxiv.org/abs/2203.02155>
- [HH22] Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback," 2022. <https://arxiv.org/abs/2204.05862>
- [UFB23] Cui et al., "UltraFeedback: Boosting Language Models with High-quality Feedback," 2023. <https://arxiv.org/abs/2310.01377>
- `trl.RewardTrainer` and `AutoModelForSequenceClassification` reference implementations. <https://github.com/huggingface/trl>
- `peft` (LoRA / QLoRA adapters). <https://github.com/huggingface/peft>
- Transformers (Hugging Face). <https://github.com/huggingface/transformers>
- Meta-Llama-3.1-8B-Instruct model card. <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>
- OWASP Machine Learning Security Top 10 (ML-02 Data Poisoning via preference pairs, ML-04 Membership Inference against the RM, ML-06 Supply Chain for pretrained base). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — reward-model attacks under "Resource Development" (preference-corpus tampering) and "Impact" (misalignment via corrupted reward). <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Measure/Manage; RM-specific bias measurement is a Measure obligation, acceptance thresholds are Govern). <https://www.nist.gov/itl/ai-risk-management-framework>
