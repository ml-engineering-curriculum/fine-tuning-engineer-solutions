# mod-104 · exercise-02 — QLoRA on a 7B Base, Single GPU — Solution

## 1. Solution overview

The deliverable is a QLoRA SFT run on a 7B base model that fits on a single consumer GPU (24 GB), with every memory-saving knob named and defended. A reviewer should be able to read `train_qlora.py`, `memory_budget.md`, and `run_report.md` and reconstruct the memory accounting from first principles — base weights, activations, optimizer state, gradients, and buffers — and see that the chosen config is the smallest that trains stably, not the one the tutorial happened to copy.

Three ideas carry the exercise:

- **QLoRA = NF4-quantized frozen base + BF16 LoRA adapters + paged optimizer** [QLoRA23]. Each piece is replaceable; dropping any one of them makes the run OOM or destabilize. The exercise is to know which one you're negotiating with when memory gets tight.
- **The memory budget is the artifact, not the config file.** A 7B base in NF4 is ~3.6 GB. Everything else — activations, grads, optimizer state, cuBLAS/attn workspace — is what you actually spend most of VRAM on.
- **BF16 compute dtype is non-negotiable on Ampere+.** FP16 QLoRA runs exist but are brittle; BF16 is what the paper and all stable configs use for forward/backward on the dequantized tensors.

This walkthrough uses `mistralai/Mistral-7B-v0.1` (Apache-2.0) as the base and a 2k-row slice of `yahma/alpaca-cleaned`. Target hardware: a single 24 GB GPU (RTX 3090 / 4090 / A5000 / L4 class). The structure applies unchanged to Llama-3 8B, Qwen-2.5 7B, or any decoder-only base of similar size.

## 2. Worked implementation

### 2.1 `train_qlora.py` — reference

```python
# train_qlora.py
from __future__ import annotations
import json, pathlib, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

BASE = "mistralai/Mistral-7B-v0.1"
OUT = pathlib.Path("out_qlora"); OUT.mkdir(exist_ok=True)

# --- 1. 4-bit quantization config ----------------------------------------
# NF4 + double-quant + bf16 compute. These are the QLoRA defaults; changing
# any of them without a reason is a memory-regression or stability bug.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    device_map={"": 0},            # single-GPU; avoid accelerate's offload heuristics
    torch_dtype=torch.bfloat16,    # buffer dtype for non-quantized tensors
    attn_implementation="sdpa",    # or "flash_attention_2" if available
)

# prepare_model_for_kbit_training does three things:
#   1. Casts layer-norm and lm_head to fp32 for numerical stability.
#   2. Enables input require_grad on the embedding (so gradient-checkpointing
#      can retain graph through non-trainable layers).
#   3. Freezes everything.
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False     # incompatible with grad-checkpointing

# --- 2. LoRA on attention + MLP projections ------------------------------
# All seven projections on Mistral: q_proj, k_proj, v_proj, o_proj,
# gate_proj, up_proj, down_proj. Attention-only (q,k,v,o) works too but
# QLoRA paper found attention+MLP matched full fine-tune quality.
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
# Expect ~42M trainable / ~7B total (~0.6%).

# --- 3. Data ------------------------------------------------------------
raw = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")
def fmt(ex):
    prompt = f"[INST] {ex['instruction']}\n{ex['input']} [/INST]"
    text = prompt + " " + ex["output"] + tok.eos_token
    enc = tok(text, truncation=True, max_length=1024, padding=False)
    return enc
ds = raw.map(fmt, remove_columns=raw.column_names)

# Causal LM collator; sets labels = input_ids, pads to longest in batch.
collator = DataCollatorForLanguageModeling(tok, mlm=False)

# --- 4. Trainer ---------------------------------------------------------
args = TrainingArguments(
    output_dir=str(OUT),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,          # effective batch 16
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_8bit",               # paged = CPU-fallback for optimizer state
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to="none",
    seed=0,
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=ds, data_collator=collator,
    tokenizer=tok,
)
trainer.train()

# Save adapter-only checkpoint (not the quantized base).
model.save_pretrained(OUT / "adapter")
tok.save_pretrained(OUT / "adapter")

# Dump the live memory stats for memory_budget.md
stats = torch.cuda.memory_stats()
(OUT / "memory_stats.json").write_text(json.dumps({
    "peak_allocated_bytes": stats.get("allocated_bytes.all.peak"),
    "peak_reserved_bytes":  stats.get("reserved_bytes.all.peak"),
}, indent=2))
```

Notes on the reference:

- **`device_map={"": 0}` not `"auto"`.** With `auto`, Transformers may offload layers to CPU to fit — the run then silently trains at CPU-RAM speed. On a 24 GB card for 7B NF4, everything fits on-GPU; pin it.
- **`prepare_model_for_kbit_training` is not optional.** It casts norms to fp32 (stability), enables input grads (for checkpointing), and freezes params. Skipping it produces NaN loss within 10 steps.
- **`gradient_checkpointing` + `use_cache=False`.** Caches are for inference and allocate per-token K/V; checkpointing re-runs forward during backward and reclaims activations. The two are incompatible by design.
- **`paged_adamw_8bit` vs. `adamw_torch`.** The paged optimizer stores moment tensors in 8-bit and *pages them to CPU* if GPU is tight (via `bitsandbytes`). For LoRA with ~42M trainable params, 8-bit AdamW state is already small (~170 MB); the paging feature matters more for full fine-tuning of 7B, but keeping it on costs nothing on this run.
- **BF16 everywhere.** `bnb_4bit_compute_dtype=bf16` means the base is dequantized to bf16 for matmul, LoRA params are bf16, optimizer state is 8-bit AdamW. FP16 will nan on long sequences; bf16 is stable [QLoRA23].
- **Attention + MLP targets.** The QLoRA paper's Table 10 showed attention-only configs hit ~98% of attention+MLP quality on MMLU but behind on code and math. For a general SFT run, target both; drop MLPs only if memory-bound.

### 2.2 `memory_budget.md` — target shape

```markdown
# QLoRA Memory Budget — Mistral-7B, seq_len=1024, bs=2, 24 GB GPU

## Per-tensor accounting (steady state, mid-training)

| Component | Count / shape | Bytes / elem | Size |
|---|---|---:|---:|
| Base weights (NF4, with double-quant) | 7.24B params in NF4 | ~0.5 | **3.62 GB** |
| Base weight buffers (dequant workspace) | peak ≈ one layer at a time | bf16 | ~0.10 GB |
| LoRA adapters (bf16, r=16, 7 projs × 32 layers) | ~42M params | 2 | **0.08 GB** |
| Adapter grads (bf16) | same shape | 2 | 0.08 GB |
| Optimizer state (paged_adamw_8bit) | 2 moments × 42M params | 1 | **0.08 GB** |
| Activations (bf16, with grad-checkpointing) | bs=2, seq=1024, hid=4096, 32 layers | 2 | **~4.5 GB** |
| Attention intermediate (SDPA / flash) | O(bs * seq * hid) + head workspace | 2 | ~1.5 GB |
| cuBLAS / allocator reserve | — | — | ~1.5 GB |
| **Total (peak)** | | | **~11.5 GB** |

Measured peak from `torch.cuda.memory_stats()`: 12.3 GB.

## What moves if you change a knob
- `seq_len` 1024 → 2048: activations grow ~2× (SDPA is O(seq²) if not using
  flash); peak roughly 15–16 GB.
- `batch_size` 2 → 4: linear in activations; peak ~17 GB.
- `gradient_checkpointing` off: activations ~2.5× larger; peak ~22 GB,
  near-OOM.
- `double_quant` off: base weights ~4.0 GB instead of 3.6 GB; marginal.
- `optim=adamw_torch` (fp32): optimizer state 4× larger (~0.3 GB); negligible
  for LoRA, but would be ~65 GB at full fine-tune, which is the whole reason
  paged_adamw_8bit exists.
- `bnb_4bit_compute_dtype=fp16`: same memory, higher nan risk on long seq.
```

### 2.3 `run_report.md` — target shape

```markdown
# QLoRA Run Report — mistral-7b-v0.1-qlora-alpaca

**Hardware:** 1× RTX 4090 (24 GB), CUDA 12.1, PyTorch 2.3, bitsandbytes 0.43,
transformers 4.44, peft 0.11.

**Config:** r=16, alpha=32, dropout=0.05, targets=qkvo+gate/up/down,
seq_len=1024, bs=2, grad_accum=8, lr=2e-4 cosine, 1 epoch, bf16, paged_adamw_8bit.

## Trainable parameters
42,467,328 / 7,283,802,112 (0.583%).

## Loss curve
step    loss   step    loss
   0   1.878     500   1.142
  50   1.421     750   1.098
 100   1.312    1000   1.067 (end epoch 1)
 200   1.221
 300   1.182

Shape is stable: fast descent for ~200 steps, then flattens; no spikes, no
NaNs. Validation loss (held-out 200 rows): start 1.95 → end 1.12.

## Peak memory
- Allocated: 12.3 GB
- Reserved:  13.9 GB
Fits with ~10 GB headroom on a 24 GB card.

## Throughput
- 1.42 s/step at effective batch 16.
- ~1,760 tokens/s forward+backward.
- Epoch wall-clock: ~24 min.

## Known issues
1. `save_steps=200` writes adapters only (~160 MB each), so disk growth is
   bounded; still, `save_total_limit=2` is cheap insurance.
2. Flash-attention 2 on 4090 is available and cuts peak memory ~20% at seq
   1024 and more at seq 2048; the run above uses SDPA for compatibility.
3. On 16 GB cards (T4, RTX 4080) this config OOMs at bs=2/seq=1024.
   Drop to bs=1/seq=512 or targets=qkv-only, and the budget halves.
```

### 2.4 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| NF4 quantization | Best-in-class 4-bit dtype for normal-distributed weights; ~0.4 bit of effective precision over int4 [QLoRA23] | int4, int8, bf16 | int4: slightly worse quality; int8: 2× the memory; bf16: 4× the memory |
| Double quantization | Quantizes the per-block scale factor itself; saves ~0.4 GB on a 7B base | Off | Cheap win; only skip if debugging bnb kernel issues |
| bf16 compute dtype | Stable over long sequences on Ampere+; QLoRA paper default | fp16 | fp16 nans on long context; fp32 doubles memory for no quality gain |
| Attention + MLP LoRA targets | Matches QLoRA paper's quality bar on general SFT | Attention-only (qkvo) | ~30% fewer trainable params, slightly worse on code/math |
| r=16, α=32 | Common operating point for 7B SFT; scale = 2.0 | r=8, r=32 | r=8 is faster/smaller but under-parameterized for MLPs; r=32 is diminishing returns on general SFT |
| Gradient checkpointing on | Activations are the dominant memory consumer at seq ≥ 1024 | Off | ~2.5× activation memory; OOM on 24 GB at seq 1024 bs 2 |
| Paged 8-bit AdamW | Zero-cost here; non-negotiable for full fine-tune of same base | adamw_torch | Fine for LoRA; blows up on full fine-tuning |
| `device_map={"": 0}` | Forces whole model to GPU 0; no silent CPU offload | `"auto"` | Auto offloads under tight memory, producing a silent 100× slowdown |
| Save adapter only | 160 MB vs. 14 GB of quantized base | Save merged model | Merging with a quantized base is not supported; must dequant first (see exercise-03) |

## 3. Validation steps

1. **VRAM check.** After the first optimizer step, `torch.cuda.max_memory_allocated() / 1e9` should land in the range `11.5–13.0 GB` for the reference config. Numbers substantially higher (e.g. 20 GB) point at missing grad-checkpointing, wrong attention impl, or an accidental fp32 cast.
2. **Trainable-param sanity.** `model.get_nb_trainable_parameters()` should print ~42M for Mistral-7B with r=16 and 7-projection targeting. If it prints more than 100M, embeddings or `lm_head` were wrapped (they shouldn't be); if less than 10M, `target_modules` didn't match the actual layer names.
3. **Loss monotonicity.** Over the first 100 steps, loss must drop monotonically on a moving average. If loss plateaus at step 10 above 2.5, `prepare_model_for_kbit_training` was skipped or dropout is too high; if it NaNs, the compute dtype is fp16 or norms are not in fp32.
4. **Attention implementation.** `print(model.config._attn_implementation)` should be `"sdpa"` or `"flash_attention_2"`. If it says `"eager"`, peak memory will be ~2× and tokens/s ~2× worse.
5. **Checkpoint shape.** `ls -lh adapter/` should show `adapter_model.safetensors` ≤ 250 MB (not 14 GB). If the base is saved, `save_pretrained` was called on the unwrapped model, not the PeftModel.
6. **Reload test.** `AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb)` + `PeftModel.from_pretrained(..., "adapter")` + `.eval()` + sample generation must produce coherent text. Catches bad adapter save or mismatched quantization at reload.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Memory budget** | No breakdown, or only reports peak VRAM | Breakdown present but missing activations / optimizer state | Per-tensor table with counts, dtypes, sizes; "what moves if you change X" section |
| **Config correctness** | Missing one of {NF4, double-quant, bf16 compute, paged optim, grad-checkpoint} | All five present but at least one unjustified | All five present, each tied to the paper or to the budget table |
| **Training stability** | NaN / diverged loss within first epoch | Converged but loss curve has unexplained spikes | Monotone loss, no NaNs, val loss drops ≥ 25% from start |
| **Reload + generate** | Adapter cannot be reloaded without the exact training session | Reloads but with manual code gymnastics | `AutoModel + PeftModel.from_pretrained` reloads cleanly; sample generation is coherent |
| **Write-up** | No report, or report doesn't tie numbers to choices | Report present, misses hardware or lib versions | Hardware, lib versions, throughput, peak memory, loss curve, known issues all present |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Using `fp16` compute dtype on BF16-capable hardware — NaN loss on long sequences, silent quality loss on shorter ones.
- Forgetting `prepare_model_for_kbit_training` — norms stay in bf16, layer-norm variance underflows, loss NaNs in <20 steps.
- Leaving `use_cache=True` with gradient checkpointing — a warning fires, cache is silently disabled, but the user often misses the warning and believes caches are helping.
- `device_map="auto"` on a single GPU — auto can offload under tight memory; the run drops to CPU-RAM throughput (~100× slower) with no error.
- Setting `optim="adamw_torch"` for a full fine-tune "to be safe" — fp32 moment tensors for 7B = ~56 GB on GPU; use paged 8-bit.
- Targeting embeddings or `lm_head` by accident (e.g. `target_modules=".*linear.*"`) — adapter is huge, memory spikes, serving becomes complicated.
- Saving the merged model instead of the adapter — merging with a quantized base requires dequantization first; the resulting file is multi-GB even though no base-weight learning happened.
- Not matching the chat template between training and eval — model learns `[INST]...[/INST]` and the eval harness sends `### Instruction:`; quality cratered by format mismatch, not by training.
- Measuring VRAM only after training finishes (post `del`) — peak memory is during the step, read `torch.cuda.max_memory_allocated()` inside the loop.
- Running flash-attention 2 on a card that doesn't support it (pre-Ampere) and silently falling back to eager — doubles peak memory and halves tokens/s without surfacing the regression.

## 6. References

- [QLoRA23] Dettmers, Pagnoni, Holtzman, Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023. <https://arxiv.org/abs/2305.14314>
- [LoRA21] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021. <https://arxiv.org/abs/2106.09685>
- `bitsandbytes` library. <https://github.com/bitsandbytes-foundation/bitsandbytes>
- `peft` documentation on `prepare_model_for_kbit_training`. <https://huggingface.co/docs/peft/>
- Transformers documentation on 4-bit loading. <https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes>
- Mistral-7B model card. <https://huggingface.co/mistralai/Mistral-7B-v0.1>
- OWASP Machine Learning Security Top 10 (ML-06 AI Supply Chain; adapter weight provenance across quantized bases). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — ML supply chain tactics, especially around model-artifact modification. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Measure/Manage — documenting training compute + quantization choices). <https://www.nist.gov/itl/ai-risk-management-framework>
