# mod-104 · exercise-01 — LoRA from Scratch and Against `peft` — Solution

## 1. Solution overview

The deliverable is a side-by-side: a from-scratch LoRA implementation that wraps `nn.Linear` layers in a small PyTorch module, and an equivalent `peft.LoraConfig` run on the same base, same data, same seed. A reviewer should be able to read `lora_from_scratch.py`, `train_scratch.py`, `train_peft.py`, and `equivalence_report.md` and agree that the two are numerically consistent — not identical bit-for-bit, but matched on loss curve, parameter count, and merged-weight delta within tolerance.

Three ideas carry the exercise:

- **LoRA is a rank-decomposed delta added to a frozen linear.** `y = x W^T + x (B A)^T * (α / r)` with `A ∈ R^{r×in}`, `B ∈ R^{out×r}`. Zero-init `B` so the adapter starts as a no-op [LoRA21].
- **The non-obvious details are initialization, scaling, and dropout placement.** Get them wrong and the loss curve looks plausible but diverges from `peft` by 5–10%.
- **"Matches `peft`" means three concrete tests.** Trainable parameter count, forward-pass equivalence at step 0, and training loss curve within tolerance over N steps.

This walkthrough uses `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` as the base (1.1B params, runs on a single 16GB GPU in fp16) and a 500-row slice of `yahma/alpaca-cleaned` for a short SFT run. The structure applies unchanged to any decoder-only base with `nn.Linear` projections.

## 2. Worked implementation

### 2.1 `lora_from_scratch.py` — the core module

```python
# lora_from_scratch.py
from __future__ import annotations
import math
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps an nn.Linear: y = x W^T + bias + (alpha/r) * x A^T B^T.

    Base weight is frozen. Only A, B, (and optional bias of the base if trainable_bias) carry grads.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        trainable_bias: bool = False,
    ):
        super().__init__()
        assert r > 0, "rank must be positive; r=0 means 'no adapter', use plain Linear"
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze the base. We keep a reference, not a copy, so merged weights
        # land back in the right tensor on merge_and_unload().
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(trainable_bias and p is self.base.bias)

        # A: [r, in_features], B: [out_features, r]
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Init: A as Kaiming-uniform (matches peft >=0.4), B as zeros so the
        # initial adapter contribution is exactly 0. This is the single most
        # common deviation; getting it wrong means step-0 forward diverges.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self._merged:
            return base_out
        # Order of ops: dropout(x) -> A -> B -> scale. Dropout is on the *input*
        # to the adapter, not on the adapter output. peft matches this.
        after_A = F.linear(self.lora_dropout(x), self.lora_A)   # [*, r]
        after_B = F.linear(after_A, self.lora_B)                # [*, out]
        return base_out + self.scaling * after_B

    @torch.no_grad()
    def merge(self) -> None:
        """Fold adapter into base weight. Idempotent, reversible via unmerge()."""
        if self._merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)      # [out, in]
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.sub_(delta.to(self.base.weight.dtype))
        self._merged = False


def _resolve(module: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> list[str]:
    """Replace every nn.Linear whose dotted name endswith one of target_modules
    with a LoRALinear wrapper. Returns the list of replaced names.

    target_modules are suffixes (e.g. ['q_proj', 'v_proj']) to match peft's
    LoraConfig(target_modules=...). Pass full dotted names if you need more control.
    """
    replaced: list[str] = []
    # Walk once; mutating during walk is fine because we replace by setattr.
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in target_modules):
            continue
        parent, attr = _resolve(model, name)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(name)

    # Freeze everything that isn't a LoRA parameter.
    for n, p in model.named_parameters():
        p.requires_grad_("lora_A" in n or "lora_B" in n)
    return replaced


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
```

Notes on the reference:

- **Init is the easy part to get wrong.** `peft` uses Kaiming-uniform with `a=sqrt(5)` on A and zeros on B (library source: `peft/tuners/lora/layer.py`'s `reset_lora_parameters`). Gaussian init on A, or non-zero B, both produce a non-trivial step-0 adapter contribution and the equivalence test fails.
- **Dropout is on the adapter *input*, not the sum.** `peft` places `nn.Dropout(dropout)` before the A-projection. If you drop the summed output, you also drop base activations through the residual — a different training dynamic.
- **Scaling is `α / r`, not `α`.** With `r=8, α=16` the scale is 2.0. Inside the forward, scaling is applied once at the end (not once per matmul). This is the convention in the original paper [LoRA21] and `peft` ≥ 0.3.
- **`merge()` / `unmerge()` must be no-ops when already in the target state.** Double-merge and you have a 2× delta baked into `base.weight`; this is a common silent bug.
- **Frozen base requires `requires_grad_(False)` everywhere except `lora_A`, `lora_B`.** If you forget and the base is fp32, gradients pile up and memory spikes.

### 2.2 `train_scratch.py` — scaffold

```python
# train_scratch.py
import json, pathlib, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from lora_from_scratch import apply_lora, count_trainable

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUT = pathlib.Path("out_scratch"); OUT.mkdir(exist_ok=True)

torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)

replaced = apply_lora(
    model,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    r=8, alpha=16, dropout=0.05,
)
trainable, total = count_trainable(model)
print(f"replaced={len(replaced)} trainable={trainable:,}/{total:,} "
      f"({100*trainable/total:.3f}%)")

# Tiny dataset: 500 alpaca-cleaned rows, template is [INST]{instruction}\n{input}[/INST]{output}
raw = load_dataset("yahma/alpaca-cleaned", split="train[:500]")
def fmt(ex):
    prompt = f"[INST] {ex['instruction']}\n{ex['input']} [/INST]"
    full = prompt + " " + ex["output"]
    enc = tok(full, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"][0]
    labels = input_ids.clone()
    prompt_ids = tok(prompt, truncation=True, max_length=512)["input_ids"]
    labels[: len(prompt_ids)] = -100
    labels[input_ids == tok.pad_token_id] = -100
    return {"input_ids": input_ids, "attention_mask": enc["attention_mask"][0], "labels": labels}

ds = raw.map(fmt, remove_columns=raw.column_names)
ds.set_format("torch")

loader = DataLoader(ds, batch_size=2, shuffle=False)  # shuffle=False for determinism
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4, weight_decay=0.0,
)
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=100)

losses = []
model.train()
for step, batch in enumerate(loader):
    if step >= 100: break
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch)
    loss = out.loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0,
    )
    opt.step(); sched.step()
    losses.append(float(loss.detach()))

(OUT / "losses.json").write_text(json.dumps(losses))
(OUT / "trainable.json").write_text(json.dumps({"trainable": trainable, "total": total}))
# Save only adapter params.
state = {n: p.detach().cpu() for n, p in model.named_parameters()
         if "lora_A" in n or "lora_B" in n}
torch.save(state, OUT / "adapter.pt")
```

### 2.3 `train_peft.py` — equivalent with `peft`

```python
# train_peft.py
import json, pathlib, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
OUT = pathlib.Path("out_peft"); OUT.mkdir(exist_ok=True)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)

cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
    init_lora_weights=True,       # Kaiming-uniform on A, zeros on B
)
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

# (identical dataset + loader + optimizer + schedule + loop as train_scratch.py)
# ... elided for brevity; use the same fmt() and the same seed ...
```

Keeping the dataset, loader, optimizer, scheduler, clipping, and seed identical is the point — the only thing that varies between the two scripts is which module wraps the linear.

### 2.4 `equivalence_report.md` — target shape

```markdown
# LoRA Equivalence: from-scratch vs. peft

## Setup
- Base: TinyLlama-1.1B-intermediate-step-1431k-3T, dtype float32, seed 0.
- Config: r=8, alpha=16, dropout=0.05, target_modules=[q,k,v,o]_proj.
- Dataset: alpaca-cleaned[:500], max_seq_length=512, batch_size=2, 100 steps.
- Optimizer: AdamW(lr=2e-4), linear warmup 5 → decay to 0 over 100 steps.

## Test 1 — Trainable parameter count
- Scratch: 4,194,304
- peft:    4,194,304
- Match: ✅ exact.

## Test 2 — Step-0 forward equivalence
With B initialized to zeros, both wrappers must produce identical logits on
the first forward because the adapter contributes nothing.

| Tensor | Max abs diff | Mean abs diff |
|---|---|---|
| logits | 0.0 | 0.0 |

Match: ✅ exact (fp32).

## Test 3 — Training loss curve (first 100 steps)

| Step | scratch loss | peft loss | abs diff |
|---:|---:|---:|---:|
|   0 | 2.413 | 2.413 | 0.000 |
|  10 | 2.198 | 2.197 | 0.001 |
|  25 | 1.874 | 1.875 | 0.001 |
|  50 | 1.612 | 1.611 | 0.001 |
|  75 | 1.498 | 1.498 | 0.000 |
|  99 | 1.441 | 1.442 | 0.001 |

- Max abs diff across all 100 steps: 0.0017.
- Tolerance: 5e-3 (fp32 AdamW over 100 steps; numerical drift accumulates slowly).
- Match: ✅ within tolerance.

(Illustrative numbers — the *shape* is what matters: both curves monotonically
decrease from ~2.4 to ~1.4 and stay within 2e-3 of each other. Exact values
depend on GPU/driver/cuBLAS version. Loss drifts > 1e-2 means a real bug,
not numerics.)

## Test 4 — Adapter weight delta
After 100 steps, compare `scaling * B @ A` between the two implementations:

| Target module | Max abs diff |
|---|---:|
| layers.0.self_attn.q_proj | 3.2e-5 |
| layers.0.self_attn.k_proj | 2.9e-5 |
| ...   | ... |
| layers.21.self_attn.o_proj | 4.1e-5 |

- Max across all modules: 7.8e-5. Tolerance: 1e-3. Match: ✅.

## Test 5 — Merged inference equivalence
Merge adapters into base, run 20 held-out prompts, compare logits:

- Max abs diff on logits: 1.4e-4.
- Top-1 token match: 20/20.
- Match: ✅.

## Where the two intentionally differ
- `peft` adds a `PeftModel` wrapper that redirects `save_pretrained` to
  adapter-only checkpoints; the scratch version has no such wrapper.
- `peft` handles 8-bit / 4-bit quantized bases out of the box (see exercise-02);
  the scratch version does not.
- `peft` carries a `modules_to_save` list for layers that must be trained in
  full (e.g. `lm_head`); the scratch version requires the user to flip
  `requires_grad_` manually.

## Conclusion
The from-scratch LoRA matches `peft` on every reproducible axis (param count,
step-0, loss curve, merged weights, merged inference) within numerical tolerance.
```

### 2.5 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Zero-init B | Adapter starts as identity delta; step-0 output matches base exactly | Small random B | Non-trivial step-0 drift; harder to prove equivalence |
| Kaiming-uniform A (a=√5) | Matches `peft`'s default; same variance scale as PyTorch `nn.Linear` reset | Gaussian(0, 1/√d) | Different output distribution at step 0; loss curves still converge, but equivalence test fails |
| Scale by α/r applied once | One multiplication; convention since [LoRA21] | Fold scaling into B at init | Scaling inside B makes LR effectively rank-dependent; annoying when sweeping r |
| Dropout on adapter *input* | Matches `peft`; dropout on base residual would change base dynamics | Dropout on sum | Breaks equivalence; also drops signal through the residual pathway |
| Suffix matching for `target_modules` | Mirrors `peft` API; one pattern covers all layers | Full dotted names | Needs more config on models whose projection names collide with other suffixes |
| Freeze everything except lora_A/lora_B | Minimal param count; stable baseline | Also train biases (`bias="all"`) | ~0.01% more params; often marginal; makes equivalence harder to prove |
| float32 baseline | Removes mixed-precision as a confound in equivalence tests | bfloat16 / fp16 | bf16 tolerance would need loosening to ~1e-2 on loss diffs |

## 3. Validation steps

1. **Trainable-count parity.** `count_trainable(scratch_model)[0] == peft_model.get_nb_trainable_parameters()[0]`. Any mismatch means a layer was wrapped by one and not the other, or the frozen set differs.
2. **Step-0 forward.** With the same input ids and `torch.manual_seed(0)` before model construction, `(scratch(x).logits - peft(x).logits).abs().max()` must be exactly 0.0 (fp32). If this fails, B is non-zero or dropout is active at step 0.
3. **Loss curve.** Train 100 steps on identical data; max abs diff across steps must stay below 5e-3 in fp32. Anything above that points at init, dropout placement, scaling, or param-group membership.
4. **Merged delta.** For each target module, `scaling * (B @ A)` from the scratch impl and `peft`'s merged delta must agree within ~1e-4. Catches off-by-one in the scaling convention (`α/r` vs. `α`).
5. **Merged inference.** `model.merge_and_unload()` on both; generate on 20 held-out prompts with the same sampling seed; top-1 tokens must agree. Catches silent double-merge bugs.
6. **Repro determinism.** Rerun on the same hardware with the same seed; losses must match bitwise in fp32 on a single GPU with `torch.use_deterministic_algorithms(True)`. (Not required, but a sanity knob when numbers drift.)

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Correctness of the LoRA module** | Wrong forward (missing scaling, wrong order, non-zero B init) | Forward is correct but merge/unmerge is buggy or non-idempotent | Forward, merge, unmerge all correct; equivalence tests pass |
| **`target_modules` targeting** | Hand-wraps a specific layer, or wraps non-Linear modules | Suffix matching works but misses GQA cases (`k_proj`/`v_proj` differ in width from `q_proj`) | Matches `peft`'s suffix semantics; handles GQA (different in/out widths on A/B) |
| **Equivalence report** | Only loss numbers, no param count or merged-delta check | Report present, missing one of the five tests | All five tests, numeric table, tolerance stated and justified |
| **Determinism of the comparison** | Seeds not set, shuffle on, dropout on across both runs — curves diverge from first step | Seeds set but something else drifts (different param-group order, mixed precision on one side) | Identical seed, dtype, loader order, optimizer config; drift ≤ 5e-3 across 100 steps |
| **Rationale / write-up** | No discussion of init, scaling, dropout placement | Mentions them but without the "why" | Each design choice tied to `peft` source or the LoRA paper; trade-offs listed |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Initializing B with small random values instead of zeros — adapter is non-trivial at step 0, equivalence test fails, loss curve drifts from `peft` by 5–10%.
- Applying `α/r` scaling inside init (folding it into B) instead of in the forward — makes LR sensitivity rank-dependent; sweeps over `r` need re-tuning.
- Placing dropout after the adapter sum — drops signal from the base residual too, changing the base model's training dynamics.
- Forgetting to freeze everything except `lora_A` / `lora_B` — optimizer updates base weights silently; memory also spikes.
- Wrapping every `nn.Linear` in the model (including `lm_head`, `embed_tokens`) — adds params, breaks tying, and is rarely what `peft` does by default.
- Double-merge: calling `merge()` twice bakes the delta in twice. Guard with a `_merged` flag.
- Comparing loss curves across fp32 and bf16 runs and claiming divergence — bf16 adds its own noise; either match dtype or loosen tolerance to ~1e-2.
- Not matching GQA widths: on Llama / Mistral with GQA, `k_proj` and `v_proj` have smaller `out_features` than `q_proj`. Each LoRA module must be sized from its own base; a shared `r × d` tensor is wrong.
- Using `shuffle=True` on the DataLoader without a Generator seeded identically on both runs — curves diverge from step 1, and it looks like a correctness bug.
- Treating `peft.print_trainable_parameters()` as ground truth without checking the from-scratch number — both can be wrong the same way if both wrap the same bad set.

## 6. References

- [LoRA21] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021. <https://arxiv.org/abs/2106.09685>
- `peft` library source (Hugging Face). <https://github.com/huggingface/peft> — `src/peft/tuners/lora/layer.py` is the authoritative implementation to diff against.
- Transformers (Hugging Face). <https://github.com/huggingface/transformers>
- TinyLlama checkpoint. <https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T>
- `yahma/alpaca-cleaned` dataset. <https://huggingface.co/datasets/yahma/alpaca-cleaned>
- OWASP Machine Learning Security Top 10 (ML-06 AI Supply Chain Vulnerabilities — PEFT library and adapter weight provenance). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — ML supply chain and model-artifact tactics. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Govern/Map/Measure/Manage; transparent documentation of training choices). <https://www.nist.gov/itl/ai-risk-management-framework>
