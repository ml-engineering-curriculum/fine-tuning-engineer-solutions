# mod-103 · exercise-03 — LR Schedule and Loss Diagnosis — Solution

## 1. Solution overview

A loss curve is a diagnostic instrument, not a vanity plot. The exercise trains four short runs on the same (base model, dataset, batch) and varies only the learning rate schedule, then asks the learner to read what each curve says and to produce a one-page diagnosis document a reviewer can act on.

Three ideas carry the exercise:

- **A loss curve has a known shape.** Warmup → steepest-descent window → long shallow tail. Deviations from this shape are named phenomena (early-step spike, floor plateau, late-run bump, grad-norm detonation, step-shaped plateau). Each one maps to a small set of causes.
- **The LR schedule is one knob with three effects.** Peak LR sets how fast you move; warmup smooths the first-steps spike; the tail (cosine vs. linear vs. constant) decides whether the model lands softly or keeps bouncing.
- **"Train loss down" is not the result.** Eval loss, grad norm, parameter-norm drift, tokens-seen, and the sampled generations together are the result. Any single one on its own is defeasible.

The worked runs use `meta-llama/Llama-3.2-1B` + `HuggingFaceH4/no_robots` (same setup as exercise-01) and vary only `learning_rate`, `lr_scheduler_type`, and `warmup_ratio`. Each run is ≤ 300 steps so a learner can produce all four on a single consumer GPU in about an hour [TRL] [L3] [NOROBOTS].

## 2. Worked implementation

### 2.1 The four runs

| Run | peak LR | scheduler | warmup_ratio | hypothesis under test |
|---|---|---|---|---|
| R1 — baseline | 2e-5 | cosine | 0.03 | "Textbook" SFT recipe. Reference curve shape. |
| R2 — too hot | 1e-4 | cosine | 0.03 | Early-step spike + grad-norm detonation from an LR above the bf16 stable band for a 1B model on this batch. |
| R3 — no warmup | 2e-5 | cosine | 0.0 | First-step spike; the rest of the curve catches up; more NaN-adjacent step. |
| R4 — constant | 2e-5 | constant | 0.03 | Late-run overshoot; eval loss turns up while train loss keeps going down. |

Every other hyperparameter is held fixed: bf16, batch 2 × 16, seq-len 2048, seed 42, weight decay 0.1, `max_grad_norm=1.0`, eval every 50 steps. Holding the other variables constant is the whole point — the curves differ only because the schedule differs.

### 2.2 `train_matrix.py` (reference)

```python
# train_matrix.py — runs R1..R4 sequentially with identical seed and data.
import os, json, pathlib, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig

MODEL_ID = "meta-llama/Llama-3.2-1B"
DATA_ID  = "HuggingFaceH4/no_robots"
OUT      = pathlib.Path("runs/lr-matrix")

RUNS = {
    "R1_baseline":   dict(learning_rate=2e-5, lr_scheduler_type="cosine",   warmup_ratio=0.03),
    "R2_too_hot":    dict(learning_rate=1e-4, lr_scheduler_type="cosine",   warmup_ratio=0.03),
    "R3_no_warmup":  dict(learning_rate=2e-5, lr_scheduler_type="cosine",   warmup_ratio=0.0 ),
    "R4_constant":   dict(learning_rate=2e-5, lr_scheduler_type="constant", warmup_ratio=0.03),
}

def load():
    ds = load_dataset(DATA_ID)
    train = ds["train_sft"] if "train_sft" in ds else ds["train"]
    eval_ = ds["test_sft"]  if "test_sft"  in ds else ds["test"]
    return train.select(range(2400)), eval_.select(range(300))  # cap for fast runs

def run(name, overrides):
    outdir = OUT / name; outdir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    train, eval_ = load()
    cfg = SFTConfig(
        output_dir=str(outdir),
        max_steps=300,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        weight_decay=0.1, max_grad_norm=1.0,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="steps", eval_steps=50,
        save_strategy="no",
        report_to=["none"],
        seed=42, data_seed=42,
        max_seq_length=2048,
        packing=False, assistant_only_loss=True,
        **overrides,
    )
    trainer = SFTTrainer(model=model, processing_class=tok, args=cfg,
                         train_dataset=train, eval_dataset=eval_)
    out = trainer.train()
    # Persist the full trainer log history: steps, loss, eval_loss, grad_norm, lr.
    (outdir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (outdir / "config.json").write_text(cfg.to_json_string())
    (outdir / "metrics.json").write_text(json.dumps(out.metrics, indent=2))

if __name__ == "__main__":
    for name, over in RUNS.items(): run(name, over)
```

### 2.3 `plot_curves.py` — the four readings

```python
# plot_curves.py — emits:
#   - loss.png (train + eval), grad_norm.png, lr.png
#   - loss.md (per-step table) for the write-up.
import json, pathlib
import matplotlib.pyplot as plt

OUT = pathlib.Path("runs/lr-matrix")

def load(run):
    hist = json.loads((OUT / run / "log_history.json").read_text())
    tr = [(h["step"], h.get("loss")) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["step"], h.get("eval_loss")) for h in hist if "eval_loss" in h]
    gn = [(h["step"], h.get("grad_norm")) for h in hist if "grad_norm" in h]
    lr = [(h["step"], h.get("learning_rate")) for h in hist if "learning_rate" in h]
    return tr, ev, gn, lr

RUNS = ["R1_baseline","R2_too_hot","R3_no_warmup","R4_constant"]

for metric, label in [("tr","train_loss"),("ev","eval_loss"),("gn","grad_norm"),("lr","lr")]:
    plt.figure()
    for r in RUNS:
        tr, ev, gn, lr = load(r)
        series = {"tr":tr,"ev":ev,"gn":gn,"lr":lr}[metric]
        xs, ys = zip(*series) if series else ([],[])
        plt.plot(xs, ys, label=r)
    plt.xlabel("step"); plt.ylabel(label); plt.legend(); plt.yscale("linear")
    plt.savefig(f"runs/lr-matrix/{label}.png", dpi=120); plt.close()
```

### 2.4 Expected curve shapes and their diagnoses

**R1 baseline (2e-5, cosine, warmup 3%).**

```
train_loss:  1.82 (s=0)  → 1.40 (s=50)  → 1.19 (s=150) → 1.08 (s=300)
eval_loss :  1.75 (s=50) → 1.30 (s=100) → 1.14 (s=200) → 1.07 (s=300)
grad_norm:   [0.9, 1.1] through warmup, then [0.3, 0.9] through tail
lr:          0 → 2e-5 by step 9 → 0 by step 300 (half-cosine)
```

Diagnosis: healthy. Train and eval drop together. Grad norm is bounded and declining as the model lands on a local minimum. Nothing to do.

**R2 too hot (1e-4, cosine, warmup 3%).**

```
train_loss:  1.82 → 2.11 (s=12 — spike) → 1.64 → NaN (s=24)
grad_norm :  1.0  → 14.3 (s=12) → 83.7 (s=18) → inf
```

Diagnosis: the peak LR is above the bf16 stable band for this model/batch/seq-len. Grad norm detonates despite `max_grad_norm=1.0` because clipping scales the direction but keeps a huge effective step at LR 1e-4 for a model whose pretraining used ~3e-4 with a billion-example warmup — the SFT budget is a thousandth of that. Remedies, in order of first-try quality: (1) drop LR to 2e-5–5e-5; (2) widen warmup to 5–10%; (3) if the task truly demands aggressive LR, add bf16 loss-scaling via an outer gradient-accumulation loop, which rarely helps as much as (1). Do not "fix" NaN by falling back to fp16 — fp16 has a tighter range than bf16 and will blow up sooner.

**R3 no warmup (2e-5, cosine, warmup 0).**

```
train_loss:  1.82 → 2.44 (s=1 — first-step spike) → 1.70 (s=10) → 1.19 → 1.08
grad_norm :  1.0  → 6.3 (s=1) → 1.2 → ...
```

Diagnosis: the first-step spike is adaptive-optimizer momentum crashing into un-warmed statistics. `AdamW`'s running moments are zero at step 0; the first step takes nearly the full peak LR as the effective step size in every direction. Warmup = 3–5% exists precisely to let `m` and `v` acquire sensible values before the LR is any larger than a rounding error. The curve recovers because the dataset is big compared to one spiked step, but under packing or small datasets, this spike can be the step that poisons the rest of training. Fix: warmup_ratio ≥ 0.03.

**R4 constant (2e-5, no decay).**

```
train_loss: 1.82 → 1.08 (s=200) → 1.00 (s=300)  — monotone down
eval_loss : 1.75 → 1.08 (s=200) → 1.12 (s=300)  — turns up
grad_norm : [0.3, 0.9] stable
```

Diagnosis: classic late-run overshoot. Without the cosine's shrinking tail, the optimizer keeps taking the same step size after it has already entered a narrow basin — it oversteps. Eval loss rises while train loss keeps dropping. This is "memorization" by another name. Fix: use cosine or linear decay to zero, or early-stop when `eval_loss` turns up for two consecutive eval points.

### 2.5 Reading the plots end-to-end

The write-up should answer *in one paragraph per plot*:

1. **Where is warmup?** On the `lr.png` plot, the warmup region should be a visible ramp. If it isn't, warmup_ratio was zero or the ratio was computed against the wrong step count.
2. **Where is the inflection?** The step where the train-loss slope becomes shallow is the inflection. For a 1B model on 2–3k rows, it usually arrives between step 100 and step 200.
3. **Train–eval gap.** A widening gap (train keeps falling, eval flattens) is capacity-not-data; a converging gap (both flat together) is data-not-capacity; a crossing gap (eval rising, train falling) is overfitting.
4. **Grad-norm shape.** Grad norm should decline slowly. A sudden order-of-magnitude bump means a batch with a wildly off-distribution example (dataset bug) or a schedule-driven detonation (R2).
5. **No NaN.** If the loss contains `NaN`, every subsequent plot is meaningless — the model is random from that step onward.

### 2.6 Rule-of-thumb LR ranges

Tie claims to published recipes rather than invented numbers:

- Llama-3 SFT recipes from third-party reproductions converge on 1e-5 – 5e-5 peak LR for full-parameter SFT at batch 32–256, bf16 [TRL] [L3].
- QLoRA SFT (mod-104) lives at 1e-4 – 3e-4 because only the adapter is trained; this does not transfer to full SFT [QLORA].
- Pretraining peaks (1e-4 – 3e-4 for 1B–7B models) are *not* valid SFT peaks; the dataset is tens-of-thousands rows, not trillions of tokens, and the warmup budget shrinks accordingly [L3] [CHINCHILLA].

If a learner picks an LR, they must also pick a warmup. The two are not independent knobs.

### 2.7 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Four runs, one variable each | Each curve attributable to one schedule choice | Grid of 16 | Grid is noisier; attribution harder for a learner |
| Hold seed/data/batch fixed | The only reason to run R2 is to see R1's curve shift | Vary data too | Confounds schedule effects with data-order effects |
| Cap at 300 steps | A reader can spot every pattern in 300 steps on this dataset | Full 2 epochs | Longer runs do not show new phenomena at this scale |
| Log every 5 steps | Enough resolution to see early-step spike (R3) | Every 50 steps | Too coarse to catch R3's step-1 spike at all |
| bf16 | Removes fp16 dynamic-range false positives | fp16 | fp16 makes R2 look worse than it is in a way that doesn't generalize |
| `max_grad_norm = 1.0` | Keeps R1/R3 clean; still fails loudly in R2 | No clipping | Without clipping, R2's step-12 blow-up is a silent corruption |

## 3. Validation steps

1. **Reproducibility of each curve.** Rerun R1 with the same seed. Max per-step `|loss_new − loss_old| < 1e-4`. If curves differ, find the non-deterministic ingredient (CUDA matmul kernel, DataLoader worker seed, FA-2 TF32).
2. **LR plot faithful.** The logged `learning_rate` at the end of warmup must equal the configured peak; at the end of a cosine-to-zero schedule, it must be within `1e-9` of zero.
3. **Grad norm logged.** `grad_norm` is `None` in some old `transformers` versions; verify the log has values. Without it, R2 diagnosis is guessing.
4. **Eval unpacked.** Confirm eval is not using packing (see mod-103 ex-02); otherwise eval numbers are not comparable across runs.
5. **NaN handling.** R2's NaN should be in the log. If the logger wrote 0.0 instead of NaN, replace the log filter — a silent 0.0 looks like success.
6. **One-page diagnosis.** The learner must produce exactly one page (≤ 600 words) naming the schedule effect for each run. Longer write-ups mean they are still exploring; shorter means they haven't looked.

## 4. Rubric for grading the learner's diagnosis

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Four-run matrix built** | Fewer than four runs, or runs vary multiple hyperparameters each | Four runs built, variables mixed in one of them | Four runs, one knob each, same seed/data, documented |
| **Curves plotted** | Only final loss numbers | Plots exist but only train loss | Train loss, eval loss, grad norm, LR, all four runs, same axes |
| **Diagnosis by name** | "Loss diverged" / "loss went up" with no causal label | At least one cause named (e.g. "LR too high") without mechanism | R2/R3/R4 each diagnosed with mechanism (dynamic range, optimizer-moment warmup, late-run overshoot) |
| **Recommended fix** | No fix, or fix is unrelated to diagnosis | Fix proposed but not justified | Fix ordered by first-try quality, cited to a published recipe or prior exercise |
| **Report + reproduce** | Ad-hoc screenshots, no command | Report exists but no `python train_matrix.py` style block | Full reproduce block, git SHA, seed, dataset hash, matched plots |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Reporting the minimum loss across the run rather than the loss at the last step. Minimum-of-noise is a lower bound no checkpoint actually achieved.
- Changing LR *and* schedule between runs. You can no longer attribute the curve shift.
- Using the same random seed but different `data_seed`. DataLoader shuffle order changes; loss comparisons become noisy.
- Taking "cosine" to mean "cosine to the peak LR minimum." In most frameworks, `cosine` decays to zero (or to `min_lr` if set); if your framework decays to peak/10 by default, your R4-vs-R1 comparison is wrong.
- "Fixing" R2's NaN by lowering `max_grad_norm`. The peak LR is the problem; clipping only hides the symptom until a bigger batch of outliers re-exposes it.
- Diagnosing R3's first-step spike as "random noise, ignore." The spike has a known cause; ignoring it means not understanding what warmup does.
- Comparing eval loss across runs with different packing or different eval batch sizes. Both change the per-row normalization.
- Averaging grad norms across steps and calling that the metric. The median and p95 matter more than the mean, which is dominated by any one spike.
- Not plotting LR. You cannot diagnose a schedule bug without seeing the schedule.

## 6. References

- [TRL] HuggingFace TRL — `SFTConfig` (`lr_scheduler_type`, `warmup_ratio`, `max_grad_norm`). <https://huggingface.co/docs/trl/main/en/sft_trainer>
- [HFTRAIN] HuggingFace Transformers — `TrainingArguments` scheduler reference. <https://huggingface.co/docs/transformers/main/en/trainer>
- [L3] Meta, "The Llama 3 Herd of Models," 2024 (fine-tuning recipe: Appendix). <https://arxiv.org/abs/2407.21783>
- [NOROBOTS] HuggingFace H4, "No Robots" dataset. <https://huggingface.co/datasets/HuggingFaceH4/no_robots>
- [ADAMW] Loshchilov & Hutter, "Decoupled Weight Decay Regularization," ICLR 2019 — AdamW moment estimates and why warmup matters. <https://arxiv.org/abs/1711.05101>
- [COSINE] Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," 2016 — cosine schedule rationale. <https://arxiv.org/abs/1608.03983>
- [CHINCHILLA] Hoffmann et al., "Training Compute-Optimal Large Language Models," 2022 — peak-LR / warmup discussion in a pretraining budget context. <https://arxiv.org/abs/2203.15556>
- [BF16] Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training," 2019 — dynamic range vs. fp16. <https://arxiv.org/abs/1905.12322>
- [QLORA] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023 — LR ranges for adapter SFT (for contrast with full SFT ranges here). <https://arxiv.org/abs/2305.14314>
- OWASP Machine Learning Security Top 10 — ML-10 Model Poisoning context (silent divergence from a hyperparameter attack). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — training-process manipulation technique category. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework — Measure/Manage; "test the process, not just the artifact." <https://www.nist.gov/itl/ai-risk-management-framework>
