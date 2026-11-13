# mod-104 · exercise-03 — Adapter Merging and Multi-Adapter Routing — Solution

## 1. Solution overview

This exercise has two halves and the point is that they are two halves, not one.

- **Merging** flattens a LoRA adapter back into the base weights so inference looks like a plain dense model. The gain is latency and simpler serving; the cost is losing the ability to swap tasks at request time.
- **Multi-adapter routing** keeps adapters separate and swaps them per-request. The gain is many tasks on one base; the cost is per-request setup overhead and a bit of extra VRAM.

The deliverable is `merge.py`, `multi_adapter_serve.py`, a `benchmarks.md` with latency/throughput numbers for both paths, and a `routing_decision.md` that names which option fits which production shape.

Three ideas carry the exercise:

- **Merging is a weight-space operation; routing is a runtime operation.** You can merge an adapter and still route by having N merged checkpoints; you can route adapters and still "merge" by caching the merged weights. The decision is latency-vs-memory, not one-vs-the-other.
- **Numerical fidelity of merge is dtype-dependent.** Merging a bf16 adapter into a NF4 base and dequantizing-for-matmul at inference is not the same as merging into a bf16 base. The exercise has to measure the divergence, not assume it's zero.
- **"Multi-adapter" means different things to vLLM, LoRAX, and PEFT alone.** The exercise picks one and explains the others.

Walkthrough base: `mistralai/Mistral-7B-v0.1` with two adapters trained from exercise-02 — one on `yahma/alpaca-cleaned` (general chat) and one on `b-mc2/sql-create-context` (text→SQL). Target: single GPU, 24 GB.

## 2. Worked implementation

### 2.1 `merge.py` — merge one adapter into the base

```python
# merge.py
from __future__ import annotations
import argparse, pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
parser.add_argument("--from-quantized", action="store_true",
                    help="If the adapter was trained on a 4-bit base (QLoRA), "
                         "load the base in full precision for the merge, not NF4.")
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
load_dtype = dtype_map[args.dtype]

if args.from_quantized:
    # IMPORTANT: Load the base in full precision for the merge even though the
    # adapter was trained on a quantized base. PEFT cannot merge into a quantized
    # weight tensor (the add_() lands in a different representation).
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=load_dtype)
else:
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=load_dtype)

model = PeftModel.from_pretrained(base, args.adapter, torch_dtype=load_dtype)
merged = model.merge_and_unload()    # folds every LoRA layer into base.weight
                                     # and strips the PeftModel wrapper.

out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"merged -> {out}")
```

Notes on the reference:

- **`merge_and_unload()` vs. `merge_adapter()`.** `merge_adapter()` folds the delta into the base weight but keeps the `PeftModel` wrapper and the LoRA parameters (they go dormant). `merge_and_unload()` folds and then strips the wrapper so you get a plain `AutoModelForCausalLM` back. Use the latter if the goal is a standalone checkpoint.
- **Quantized merge is not supported.** You cannot call `merge_and_unload()` on a model loaded with `BitsAndBytesConfig(load_in_4bit=True)`; the add-into-NF4 is undefined. Load the base in bf16/fp32 for the merge, then re-quantize at serving time if you need 4-bit inference.
- **Dtype choice is a fidelity choice.** Merging in bf16 and serving in bf16 is the common case and introduces ~1e-4 logit diff vs. the unmerged adapter path. Merging in fp32 and downcasting to bf16 at save time is closer to the "ideal" delta (~5e-5) but doubles merge-time memory.
- **Tokenizer tags along.** Saving only the model and reloading with a different tokenizer breaks the chat template; always co-save.

### 2.2 `multi_adapter_serve.py` — switch adapters at request time

```python
# multi_adapter_serve.py — PEFT-only serving, single base + N adapters.
from __future__ import annotations
import pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "mistralai/Mistral-7B-v0.1"
ADAPTERS = {
    "chat": "out_qlora_chat/adapter",
    "sql":  "out_qlora_sql/adapter",
}

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16,
)

# Load the first adapter under its name, then add the rest.
model = PeftModel.from_pretrained(base, ADAPTERS["chat"], adapter_name="chat")
model.load_adapter(ADAPTERS["sql"], adapter_name="sql")

@torch.inference_mode()
def generate(prompt: str, adapter: str, max_new_tokens: int = 128) -> str:
    assert adapter in ADAPTERS, f"unknown adapter {adapter}"
    model.set_adapter(adapter)                 # <-- the swap
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

# Minimal dispatch: a tiny classifier or rule on the prompt picks the adapter.
def route(prompt: str) -> str:
    low = prompt.lower()
    if any(k in low for k in ["select ", "from ", " where ", "sql"]):
        return "sql"
    return "chat"

if __name__ == "__main__":
    for p in [
        "Explain monads to a bored product manager.",
        "Generate SQL: list customers who placed > 3 orders last month.",
    ]:
        a = route(p)
        print(f"[{a}] {p}\n{generate(p, a)}\n---")
```

Notes on the reference:

- **`set_adapter` is a reference swap, not a weight copy.** PEFT keeps all loaded adapter tensors in VRAM; the active one is selected by swapping pointers inside each LoRA layer. Swap cost is O(num_lora_layers), not O(params).
- **The base is loaded once.** All adapters share the quantized base; adding a second adapter adds only `~170 MB` for 7B at r=16. This is the main memory argument for routing over merging-N-copies.
- **Routing logic belongs outside the model.** The example uses a tiny rule for brevity, but real systems use: a header set by the gateway, a tenant ID lookup, or a prompt classifier. The model should not infer which adapter to use from the prompt content.
- **PEFT's `set_adapter` is single-adapter-per-call.** `model.add_weighted_adapter(...)` can blend two adapters with weights, which is a different and more fragile feature (see §2.4).

### 2.3 `benchmarks.md` — target shape

```markdown
# Benchmarks — Merged vs. Multi-Adapter Serving

**Hardware:** 1× RTX 4090 (24 GB), seq=512 prompt / 128 new tokens, bs=1,
greedy decode, 50 prompts per condition.

## Latency (p50 / p95, ms per request)

| Serving path | Base dtype | p50 | p95 | tokens/s |
|---|---|---:|---:|---:|
| Base, no adapter (bf16) | bf16 | 2,180 | 2,310 | 58.7 |
| Base, no adapter (NF4) | NF4+bf16 | 2,520 | 2,660 | 50.8 |
| Merged (bf16) | bf16 | 2,180 | 2,310 | 58.7 |
| Merged (NF4, re-quantized after merge) | NF4+bf16 | 2,540 | 2,680 | 50.4 |
| LoRA adapter active (bf16) | bf16 | 2,260 | 2,390 | 56.6 |
| LoRA adapter active (NF4) | NF4+bf16 | 2,610 | 2,770 | 49.0 |
| Adapter swap on first call after `set_adapter` | NF4+bf16 | 2,640 | 2,810 | 48.6 |

- Adapter overhead vs. merged at the same base dtype: ~3–4% latency.
- First call after `set_adapter` adds ~30 ms (one-time pointer swap);
  subsequent calls hit the second row.

## Memory (steady state)

| Setup | VRAM |
|---|---:|
| Base NF4 only | 4.8 GB |
| Base NF4 + 1 adapter loaded | 4.96 GB |
| Base NF4 + 2 adapters loaded, active=chat | 5.12 GB |
| 2× merged checkpoints (chat + sql), one loaded | 14.6 GB (per model) × 2 if both resident |

Routing wins decisively on memory when you need ≥ 2 tasks on one GPU.

## Numerical fidelity (merged vs. unmerged on the same adapter)

| Metric (50 prompts) | Diff |
|---|---|
| Max abs logit diff | 1.6e-4 |
| Top-1 token match | 50/50 |
| Top-5 token overlap | 50/50 |

No observed behavioral divergence between merged and unmerged paths at
bf16. Merging while base is NF4 is not evaluated here because PEFT refuses
the operation (by design).
```

### 2.4 `routing_decision.md` — target shape

```markdown
# Routing Decision — When to Merge, When to Route

## Rule of thumb
- **1 adapter in production, latency-sensitive →** merge. The ~3% latency
  savings and the simpler serving (no PEFT at runtime) win.
- **2–10 adapters, moderate QPS →** route with PEFT or LoRAX. Base is
  shared, memory is near-flat with adapter count.
- **>10 adapters, high QPS →** vLLM with `--enable-lora`. Batched requests
  with different adapters share kernels; this is the only path that scales
  past a handful of adapters [vLLM-LoRA].

## When merging is wrong
- You still need to swap behaviors at request time (per-tenant personas,
  per-task formats). Merging commits you to one set of weights per process.
- The base is quantized and you want to ship the quantized model. Merging
  requires dequantizing first; the resulting merged checkpoint is multi-GB
  and the re-quantization introduces a second round of error.

## When routing is wrong
- Single-adapter workload where you also need the absolute lowest latency
  (sub-40ms TTFT territory). Three percent matters there; the PEFT wrapper
  adds a wrapped `forward` call per linear.
- Air-gapped or edge deployments where "two files on disk" is operationally
  simpler than "one base + N adapter dirs."

## When neither fits
- Blending adapters at inference (combine `style-formal` with `domain-legal`
  at weights 0.7/0.3). PEFT's `add_weighted_adapter` supports this but is
  fragile: linear combinations of LoRA deltas are not rigorously grounded,
  and the only reliable use case is "two closely-related task adapters,
  weights summing to ~1" [PEFT-WA]. Treat as experimental.

## Decision at a glance
| Adapters | Latency budget | Memory budget | Choice |
|---|---|---|---|
| 1 | Tight (< 50 ms TTFT) | Not constrained | Merge |
| 1 | Loose (> 100 ms TTFT) | Tight | Route (adapter only) |
| 2-10 | Any | Moderate | Route with PEFT |
| 10+ | Any | Any | Route with vLLM `--enable-lora` or LoRAX |
```

### 2.5 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| `merge_and_unload` over `merge_adapter` | Strips the PEFT wrapper; result is a plain `AutoModelForCausalLM` suitable for any serving stack | `merge_adapter` | Keeps the wrapper so you could unmerge; useful during iteration, wrong for production checkpoints |
| Dequantize-for-merge when base is NF4 | PEFT cannot merge into NF4 weight tensors | Try to merge in place and fail | A silent fail path doesn't exist; the attempt will raise, which is good |
| `set_adapter` string API over `add_weighted_adapter` | Simple, unambiguous, O(layers) swap | Blended adapters | Blending is attractive but fragile; a small quality win most days, a big quality regression the day a tenant's prompt happens to hit an unstable interpolation point |
| Single-GPU multi-adapter via PEFT | Fits the exercise scope; honest about the ~3% latency cost | vLLM + `--enable-lora` | vLLM is the correct answer in production at scale, but the setup eclipses the exercise |
| Prompt-side routing (rule or classifier) | Keeps responsibility at the gateway; model stays a plain function | Prompt-conditioned routing inside the model | Mixing retrieval of the right adapter with the model under test makes evals hard to attribute |
| Co-save tokenizer with merged model | Chat templates differ; re-training a merged ckpt with the wrong template silently degrades quality | Reload tokenizer at serve time from the base ID | Base ID may rotate; shipping the exact tokenizer bytes is the safe default |

## 3. Validation steps

1. **Merge equivalence.** For 50 held-out prompts, compare greedy decoded outputs from `PeftModel(base, adapter).generate(...)` vs. `merged_model.generate(...)`. Top-1 tokens must agree on all 50 under bf16; logit max-abs diff ≤ 5e-4.
2. **Idempotency.** Calling `merge_and_unload()` on an already-merged model is a programmer error (the wrapper is gone); but on a `merge_adapter()` result, calling `unmerge_adapter()` must restore logits to within 1e-5 of the unmerged version. Catches double-merge drift.
3. **Adapter swap latency.** Time `model.set_adapter("sql")` followed by one forward pass. First call: ~20–40 ms extra; subsequent calls: ≤ 2 ms extra over base forward. If steady-state swap cost is > 50 ms, something is re-materializing weights.
4. **Memory flatness in routing.** Load N adapters; `torch.cuda.memory_allocated()` should grow by roughly `n_layers × (in+out) × r × 2 bytes` per adapter (≈170 MB for 7B at r=16). Growth proportional to the quantized base size means an adapter is duplicating the base.
5. **Routing correctness.** Send 20 chat prompts and 20 SQL prompts through the routing function; log which adapter answered. 100% hit rate for the trivial rule, or ≥ 95% for the classifier variant. Below that, the rule is sending out-of-domain traffic to the wrong adapter and the quality numbers are meaningless.
6. **Quantized-serving round-trip.** Merge in bf16, re-quantize to NF4 for serving, generate the same 50 prompts. Top-1 match vs. bf16-merged on ≥ 45/50 is acceptable; below that, one of the quantization steps lost structure and you should either stay in bf16 or use a per-channel scheme.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Merge correctness** | Merges into NF4 silently, or ships a merged ckpt that doesn't reload | Merge works but tokenizer / chat template drifts | Merge + reload + generation parity vs. unmerged on 50 prompts |
| **Multi-adapter serving** | Reloads full model to switch adapters | Routes via `set_adapter` but with per-call ~100 ms overhead | Swap ≤ 2 ms steady state; memory grows linearly in adapter count, not model count |
| **Benchmarks** | Latency only, or one condition only | Latency across merged vs. routed, but missing memory | Latency + memory + numerical fidelity across all four relevant conditions (bf16/NF4 × merged/routed) |
| **Routing decision write-up** | "Just merge" or "just route" with no conditions | Picks one and mentions the alternative in passing | Decision table conditioned on #adapters × latency × memory; names when neither fits |
| **Numerical fidelity analysis** | No logit diff, no top-1 check | Top-1 only | Top-1 + logit max-abs + behavioral regression check across conditions |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Calling `merge_and_unload()` on a 4-bit-loaded model and catching the exception to "retry" — the retry doesn't help; load the base in bf16 for the merge and re-quantize later if needed.
- Saving the merged model without the tokenizer — downstream reload picks up the base tokenizer, which may have a slightly different chat template, and the model quietly underperforms.
- Treating `set_adapter("name")` as free — the first call after swap has a one-time cost (pointer updates, cache invalidation). Warm up each adapter before benchmarking.
- Measuring latency at bs=1 and extrapolating to bs=32 — multi-adapter kernels in PEFT don't batch across adapters; vLLM's `--enable-lora` does. The scaling story is different.
- Blending adapters with `add_weighted_adapter` and claiming behavior parity with either source — interpolated LoRA deltas are not linear in behavior space; evaluate the blend as a separate model.
- Assuming "merged = faster". It is, but only by 2–5% on single-adapter inference; the operational cost (one checkpoint per task, no runtime swap) usually outweighs the win once you have more than one task.
- Forgetting to set `model.config.use_cache = True` for inference after training — leaves grad-checkpointing on, decode becomes ~3× slower.
- Routing by inspecting the prompt in the model layer — makes evals non-reproducible (route depends on prompt which depends on tokenizer which depends on model). Keep routing at the gateway.
- Shipping a merged NF4 checkpoint and listing it as a "4-bit model" when the merge involved bf16 dequantization — the file is fp16 or bf16 safetensors; re-quantize at load time or state the dtype honestly.
- Benchmarking with `do_sample=True` and seeing variance in logits; disable sampling for determinism in the merge-fidelity test.

## 6. References

- [LoRA21] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021. <https://arxiv.org/abs/2106.09685>
- [QLoRA23] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," 2023. <https://arxiv.org/abs/2305.14314>
- [PEFT-WA] `peft` documentation, `PeftModel.add_weighted_adapter`. <https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftModel.add_weighted_adapter>
- [vLLM-LoRA] vLLM multi-LoRA serving docs. <https://docs.vllm.ai/en/latest/features/lora.html>
- LoRAX (Predibase) multi-adapter inference server. <https://github.com/predibase/lorax>
- `peft` source, `merge_and_unload` implementation. <https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/model.py>
- Transformers documentation. <https://huggingface.co/docs/transformers>
- OWASP Machine Learning Security Top 10 (ML-06 AI Supply Chain; merged weights inherit all upstream adapter provenance obligations). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — model-artifact modification and multi-tenant serving considerations. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Govern/Measure — documenting which tenant hits which adapter is auditable behavior). <https://www.nist.gov/itl/ai-risk-management-framework>
