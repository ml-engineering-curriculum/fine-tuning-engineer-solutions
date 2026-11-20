# mod-109 · exercise-01 — GPTQ vs. AWQ vs. bitsandbytes 4-bit Comparison — Solution

## 1. Solution overview

The deliverable is a **three-way, apples-to-apples comparison** of three 4-bit post-training quantization (PTQ) strategies applied to the *same* fine-tuned checkpoint, followed by a one-page recommendation. The three strategies are:

- **GPTQ** — weight-only PTQ that minimizes layer-wise reconstruction error on a small calibration set, using a Hessian-based update (Frantar et al., *GPTQ*, 2023). Produced offline with `auto-gptq` / `gptq-for-llamas` or the Transformers-integrated `GPTQModel` / `optimum` paths.
- **AWQ** — activation-aware weight quantization that identifies the salient 1% of weights and protects them by scaling so the per-channel outliers survive the INT4 round trip (Lin et al., *AWQ*, 2023). Produced offline with `llm-awq` / `autoawq`.
- **bitsandbytes 4-bit (NF4)** — runtime weight quantization used by QLoRA (Dettmers et al., *QLoRA*, 2023). No calibration pass; weights are dequantized just before the matmul on the fly, via the bitsandbytes CUDA kernels wired into Transformers (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`).

Four ideas carry the exercise:

- **Apples-to-apples means one checkpoint, one eval set, one hardware fixture.** Different base checkpoints, different `lm-eval-harness` task versions, different GPUs, or different inference runtimes make the deltas unreadable. Quantize the same fine-tuned checkpoint three ways, run the same eval harness and the same throughput harness against each, and keep the uncompressed bf16 model as the ground-truth row.
- **Accuracy, memory, latency, and *build cost* are all axes.** GPTQ and AWQ require a one-time calibration pass (minutes to a few hours); bitsandbytes has no calibration cost but typically slower kernels. A recommendation that reports accuracy alone has only looked at one corner of the design space.
- **Measure both prefill and decode.** 4-bit weights shrink the memory footprint and move decode (memory-bandwidth-bound) toward "fast"; they do little or nothing for prefill (compute-bound) when the activations stay in fp16/bf16. A single tok/s number averaged over prefill+decode hides this. Report prompt-processing tok/s and generation tok/s separately.
- **Format and runtime are coupled.** GPTQ weights serialize to a specific kernel contract (e.g. `exllamav2`, `marlin`, `gptq` kernels); AWQ weights deserialize into their matching kernels (`awq`, `awq_marlin`); NF4 weights round-trip through bitsandbytes at runtime. "Can vLLM load this?" is a format question, not a quantization-quality question, and must be settled before the exercise starts.

The worked example quantizes `./runs/llama-3.2-3b-sft` (the SFT output used throughout mod-108) three ways on 1× A100-80GB, evaluates each on the frozen mod-108 task set (`mmlu`, `arc_challenge`, `hellaswag`, `winogrande`, `truthfulqa_mc2`, `gsm8k`, and the `internal_task_v1` custom task), and measures prefill/decode throughput on vLLM for the two offline formats and on `transformers` for the bitsandbytes variant.

## 2. Worked implementation

### 2.1 Environment and pins

```text
# requirements.txt — pin everything; kernel ABIs drift between minor versions
torch==2.4.1
transformers==4.45.2
accelerate==1.0.1
datasets==3.0.1
sentencepiece==0.2.0

# Offline PTQ toolchains
auto-gptq==0.7.1                  # produces GPTQ weights
autoawq==0.2.6                    # produces AWQ weights
optimum==1.23.1                   # wrappers + exporters

# Runtime quant
bitsandbytes==0.44.1              # NF4 / INT4 on-the-fly

# Serving + eval
vllm==0.6.3                       # loads gptq, awq, and bf16 — not bnb
lm-eval==0.4.5                    # consistent with mod-108
```

Record `pip freeze` and `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv` in `env.txt`.

### 2.2 Calibration data for GPTQ and AWQ

Both methods need a calibration set that matches the deployment distribution. Using wiki-text when deploying a code model gives low perplexity on wiki-text and worse downstream accuracy than calibrating on code. Rule of thumb: 128–512 samples of 2,048 tokens each, sampled from the *post-fine-tune* evaluation distribution, not the pretraining corpus.

```python
# calib.py — materialize a small calibration set for both GPTQ and AWQ
import json, random, pathlib
from datasets import load_dataset

random.seed(1234)

# Mix: 60% task-distribution (what the model will actually see at inference), 40% a
# broad-coverage dataset so sparse layers see a variety of activations.
task_ds = load_dataset("json", data_files="./data/sft_eval_prompts.jsonl", split="train")
broad_ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

samples = [r["text"] for r in task_ds.select(range(300))]
for i, row in enumerate(broad_ds):
    if i >= 200:
        break
    samples.append(row["text"])

random.shuffle(samples)
pathlib.Path("calib.jsonl").write_text(
    "\n".join(json.dumps({"text": s}) for s in samples[:512]) + "\n"
)
```

### 2.3 Quantize with GPTQ (`auto-gptq`)

```python
# quant_gptq.py — produces outputs/gptq/ with safetensors + config
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import json, pathlib

MODEL = "./runs/llama-3.2-3b-sft"
OUT = pathlib.Path("outputs/gptq")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
examples = [
    tokenizer(json.loads(line)["text"], return_tensors="pt", truncation=True, max_length=2048)
    for line in pathlib.Path("calib.jsonl").read_text().splitlines()[:128]
]

quant_cfg = BaseQuantizeConfig(
    bits=4,
    group_size=128,         # 128 is the standard; smaller => higher accuracy, bigger file
    desc_act=True,          # activation-order reordering; small accuracy win, slight kernel slowdown
    damp_percent=0.01,
    sym=True,
)

model = AutoGPTQForCausalLM.from_pretrained(MODEL, quantize_config=quant_cfg, torch_dtype="auto")
model.quantize(examples, batch_size=1, use_triton=False)
model.save_quantized(str(OUT), use_safetensors=True)
tokenizer.save_pretrained(str(OUT))
```

Record the wall-clock time; on a 3B model with 128 examples this is typically 15–40 minutes on one A100. The `desc_act=True` choice gives slightly better accuracy at a small decode-latency cost under the `exllamav2` kernel; flip to `False` if latency is tighter than accuracy and defend the trade in the report.

### 2.4 Quantize with AWQ (`autoawq`)

```python
# quant_awq.py — produces outputs/awq/ with safetensors + config
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import json, pathlib

MODEL = "./runs/llama-3.2-3b-sft"
OUT = pathlib.Path("outputs/awq")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
calib = [json.loads(line)["text"] for line in pathlib.Path("calib.jsonl").read_text().splitlines()[:128]]

model = AutoAWQForCausalLM.from_pretrained(MODEL, device_map="cuda", safetensors=True)
model.quantize(
    tokenizer=tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,   # match the GPTQ run for an honest comparison
        "w_bit": 4,
        "version": "GEMM",     # "GEMM" is the vLLM-compatible packing; "GEMV" is batch-1-faster
    },
    calib_data=calib,
    max_calib_samples=128,
    max_calib_seq_len=2048,
)
model.save_quantized(str(OUT), safetensors=True)
tokenizer.save_pretrained(str(OUT))
```

`version="GEMM"` is required if the model is served under vLLM — the `awq` and `awq_marlin` kernels assume GEMM packing. If the deployment target is standalone `autoawq` single-stream batch-1 (e.g. a local workstation), `GEMV` is 10–30% faster at batch 1 but is not the production layout.

### 2.5 "Quantize" with bitsandbytes (no offline step)

bitsandbytes does not produce a quantized artifact on disk — the fp16/bf16 weights are loaded, then packed to NF4 by the kernel at model-load time. The "config" to record is the `BitsAndBytesConfig`:

```python
# load_bnb.py — reference loader used by lm-eval and the throughput harness
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "./runs/llama-3.2-3b-sft"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # "nf4" is the QLoRA default; "fp4" is the alternative
    bnb_4bit_use_double_quant=True,       # ~0.4 bits/param saved; negligible quality impact
    bnb_4bit_compute_dtype=torch.bfloat16, # activations/matmul remain in bf16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_cfg, device_map="auto")
```

Document the config in `report.md`. "bitsandbytes NF4 + double-quant, compute dtype bf16" is the full specification of this row.

### 2.6 Baseline: the unquantized bf16 checkpoint

The three quantized variants need a reference row. Load the SFT checkpoint at bf16 with no quant and run the same two harnesses (accuracy + throughput). The baseline anchors the quality-delta column and the memory-saved column.

### 2.7 Accuracy evaluation (reuses mod-108 harness)

```bash
# eval_all.sh — one pass per variant, same seeds and task set as mod-108/exercise-01
set -euo pipefail

VARIANTS=(
  "bf16   ./runs/llama-3.2-3b-sft           hf       pretrained={m},dtype=bfloat16"
  "gptq   ./outputs/gptq                    hf       pretrained={m},dtype=bfloat16"
  "awq    ./outputs/awq                     hf       pretrained={m},dtype=bfloat16"
  "bnb    ./runs/llama-3.2-3b-sft           hf       pretrained={m},dtype=bfloat16,load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=bfloat16"
)

for row in "${VARIANTS[@]}"; do
  read -r name m runner args <<<"$row"
  args=${args//\{m\}/$m}
  mkdir -p "results/${name}"
  lm_eval \
    --model "$runner" \
    --model_args "$args" \
    --tasks mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k,internal_task_v1 \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --seed 1234 --fewshot_random_seed 1234 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --output_path "results/${name}" \
    --log_samples
done
```

Reuse the mod-108 `compare.py` to produce a four-column table (base-bf16, gptq, awq, bnb). The z-test logic carries over — the only change is that every quantized row is compared against the *bf16 variant of the same checkpoint*, not against the pretrained base.

### 2.8 Throughput: prefill and decode, separately

Four runs, one harness. For GPTQ and AWQ, use vLLM so the production kernels are exercised; for bitsandbytes, use `transformers.generate` under its CUDA kernels because vLLM does not load bnb weights.

```python
# throughput.py — prefill tok/s and decode tok/s, measured independently
from __future__ import annotations
import argparse, json, time, statistics, pathlib

def bench_vllm(model_dir: str, quant: str | None, prompt_tokens: int, gen_tokens: int, n: int = 32):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_dir, quantization=quant, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096, enforce_eager=False)
    prompt = "x " * prompt_tokens
    params = SamplingParams(max_tokens=gen_tokens, temperature=0.0, ignore_eos=True)
    # Warmup: compile + cudagraph capture
    _ = llm.generate([prompt] * 2, params)
    # Prefill-only: generate 1 token so wall-clock is dominated by prefill
    prefill_params = SamplingParams(max_tokens=1, temperature=0.0)
    pt = _time(lambda: llm.generate([prompt] * n, prefill_params))
    prefill_tps = (n * prompt_tokens) / pt
    # Prefill+decode: subtract prefill to isolate decode rate
    full = _time(lambda: llm.generate([prompt] * n, params))
    decode_tps = (n * gen_tokens) / max(1e-6, full - pt)
    return {"prefill_tok_per_s": prefill_tps, "decode_tok_per_s": decode_tps}

def bench_hf_bnb(model_dir: str, prompt_tokens: int, gen_tokens: int, n: int = 32):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=cfg, device_map="auto")
    m.eval()
    ids = tok(["x " * prompt_tokens] * n, return_tensors="pt", padding=True).to(m.device)
    torch.cuda.synchronize()
    # Prefill-only: 1 generated token
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = m.generate(**ids, max_new_tokens=1, do_sample=False, use_cache=True)
    torch.cuda.synchronize(); pt = time.perf_counter() - t0
    # Full
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = m.generate(**ids, max_new_tokens=gen_tokens, do_sample=False, use_cache=True)
    torch.cuda.synchronize(); full = time.perf_counter() - t0
    return {"prefill_tok_per_s": (n * prompt_tokens) / pt,
            "decode_tok_per_s": (n * gen_tokens) / max(1e-6, full - pt)}

def _time(fn):
    import torch; torch.cuda.synchronize()
    t0 = time.perf_counter(); fn()
    torch.cuda.synchronize(); return time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name:runner:model_dir:quant (quant may be empty)")
    ap.add_argument("--prompt_tokens", type=int, default=512)
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    out = {}
    for spec in args.runs:
        name, runner, model_dir, quant = spec.split(":")
        quant = quant or None
        if runner == "vllm":
            out[name] = bench_vllm(model_dir, quant, args.prompt_tokens, args.gen_tokens, args.n)
        elif runner == "hf":
            out[name] = bench_hf_bnb(model_dir, args.prompt_tokens, args.gen_tokens, args.n)
    out["_config"] = vars(args)
    pathlib.Path("results/throughput.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

Invocation:

```bash
python throughput.py \
  --runs \
    bf16:vllm:./runs/llama-3.2-3b-sft: \
    gptq:vllm:./outputs/gptq:gptq \
    awq:vllm:./outputs/awq:awq \
    bnb:hf:./runs/llama-3.2-3b-sft: \
  --prompt_tokens 512 --gen_tokens 128 --n 32
```

Record `nvidia-smi --query-gpu=memory.used --format=csv` at steady state during each run and attach to `report.md`.

### 2.9 Worked summary table (illustrative; numbers depend on hardware and model)

Reference figures from running the pipeline above on 1× A100-80GB against the mod-108 SFT checkpoint. Treat the digits as illustrative; the examinable claims are the *direction* and *order of magnitude* of each delta.

| Variant | On-disk GB | Steady VRAM | MMLU | internal_v1 | Prefill tok/s | Decode tok/s | Build time |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 (baseline)            | 6.4 | 7.8 GB | 0.526 | 0.731 | 18,400 | 148 | — |
| GPTQ 4-bit g128 desc_act   | 2.0 | 3.1 GB | 0.518 | 0.724 | 14,200 | 258 | 28 min |
| AWQ 4-bit g128 GEMM        | 2.0 | 3.2 GB | 0.522 | 0.728 | 15,100 | 272 | 41 min |
| bnb NF4 double-quant       | 2.1 | 3.4 GB | 0.515 | 0.719 |  5,300 |  94 | 0 min |

Pattern to read:

- **Accuracy: all three stay within ~1 point MMLU of bf16, with AWQ consistently closest.** This matches the published AWQ result that outlier-aware scaling preserves instruction-tuned-model quality better than pure error-minimizing PTQ.
- **Decode throughput: GPTQ and AWQ win on vLLM.** The 4-bit weight fetch turns memory bandwidth into less of a bottleneck in the decode loop; prefill is largely unchanged because matmuls still run at bf16 compute.
- **bitsandbytes is slower end-to-end** because the dequantize-into-registers path is not fused into the Transformers matmul kernels the way GPTQ/AWQ kernels are in vLLM. bnb's advantage is zero build time and no format pinning — ideal for training-time use (QLoRA) and ad-hoc inference, not for throughput-sensitive serving.
- **Build cost is real**. On a model where the fine-tune pipeline reships every week, 30-40 minutes of GPU time per release for GPTQ/AWQ calibration must be scheduled; bnb has zero.

### 2.10 Directory layout (as delivered)

```
outputs/exercise-01/
├── env.txt
├── calib.jsonl
├── quant_gptq.py
├── quant_awq.py
├── load_bnb.py
├── eval_all.sh
├── throughput.py
├── outputs/
│   ├── gptq/              # safetensors + quantize_config.json
│   └── awq/               # safetensors + quantize_config.json
├── results/
│   ├── bf16/results_<ts>.json
│   ├── gptq/results_<ts>.json
│   ├── awq/results_<ts>.json
│   ├── bnb/results_<ts>.json
│   ├── throughput.json
│   └── comparison.md
└── report.md
```

### 2.11 `report.md` skeleton

```markdown
# GPTQ vs. AWQ vs. bitsandbytes — report

## Setup
- Base checkpoint: ./runs/llama-3.2-3b-sft (mod-103 output)
- Hardware: 1× A100-80GB; CUDA 12.4; bf16 compute
- Harness: lm-evaluation-harness v0.4.5 (same task YAML as mod-108/exercise-01)
- Calibration set: 512 rows (60% task distribution + 40% C4-en), seqlen 2048
- Serving: vLLM 0.6.3 for bf16/gptq/awq; Transformers 4.45.2 + bitsandbytes 0.44.1 for bnb
- Date: 2026-11-20

## Results
(see table in §2.9 and comparison.md)

## Decision
For this checkpoint and workload: **AWQ 4-bit g128 GEMM**.
- Within ~0.4 pts of bf16 MMLU, matches internal_v1 to within noise.
- Highest decode tok/s of the three; vLLM-native kernel.
- 41-minute build cost is acceptable for the weekly release cadence.

Runner-up: GPTQ. Pick it instead if the calibration pipeline is already
instrumented for GPTQ or if a non-vLLM runtime (e.g. ExLlamaV2) is a target.

Reject bitsandbytes for serving; retain it for ad-hoc inference and QLoRA training.

## Risks and follow-ups
1. Calibration-distribution drift: if the deployed traffic shifts away from the
   task-eval distribution, re-run calibration. Track with a periodic KL between
   production prompt embeddings and calibration prompt embeddings.
2. Kernel pinning: both AWQ and GPTQ require a specific vLLM kernel
   (`awq_marlin` / `gptq_marlin`); bump the vLLM version only alongside a
   repeat of this harness, not on the fly.
3. 3B model only: larger models can show larger quant deltas, especially on
   math (GSM8K) and reasoning. The decision above should be re-validated when
   the checkpoint moves to 7B or 70B.
```

## 3. Validation steps

1. **Calibration set is sealed.** `calib.jsonl` has a sha256 recorded in `report.md`, and both `quant_gptq.py` and `quant_awq.py` read the first 128 lines deterministically. Re-running the two quantizers produces byte-identical safetensors (up to kernel-level nondeterminism, which is logged).
2. **All four accuracy runs share the same task YAML, seeds, and chat-template decision.** `results/*/results_*.json` show identical `versions` blocks and identical `fewshot_seed` / `numpy_seed`. Any divergence invalidates the comparison row.
3. **Every row has stderr and CI95.** The `compare.py` harness from mod-108 is reused as-is; the "verdict" column is driven by the two-proportion z-test, not by raw point estimates.
4. **Throughput is measured prefill-separately.** `throughput.json` shows `prefill_tok_per_s` and `decode_tok_per_s` as independent numbers for every variant. A single "tok/s" column is a red flag — reject the submission and ask for the split.
5. **Memory is measured at steady state, not at load time.** VRAM is sampled after warmup with `nvidia-smi --query-gpu=memory.used --format=csv -l 1` and the median is recorded; loading-time spikes are explicitly excluded.
6. **Kernel path is named.** `report.md` states which vLLM kernel flag was used (`--quantization awq`, `--quantization gptq`, or `--quantization awq_marlin` / `gptq_marlin`) and the bnb config. "4-bit" with no kernel name is not a specification.
7. **Build time is logged.** The wall-clock for the GPTQ and AWQ calibration passes is recorded in `report.md`. "Zero" is the correct number for bitsandbytes.
8. **No-regression rule on the custom task.** `internal_task_v1` is the task the fine-tune was *for*; a quant variant that moves it by more than 2 standard errors is flagged as a loss regardless of its MMLU number.
9. **The baseline row exists and is from the same checkpoint.** It is not valid to compare a quantized SFT model to a quantized pretrained base; the bf16 baseline is `./runs/llama-3.2-3b-sft` loaded at full precision.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Apples-to-apples setup** | Different checkpoints or different eval sets across variants | Same checkpoint and tasks but seeds or chat-template flags diverge between runs | Identical checkpoint, task YAML, seeds, and chat-template decision; env captured; pins recorded |
| **Three variants produced and loadable** | One or more variants fails to load or evaluate | All three produced but one does not run under its documented runtime | All three load and evaluate; GPTQ/AWQ serve under vLLM; bnb serves under Transformers; configs recorded |
| **Accuracy comparison** | Point deltas without CIs; no reference to the bf16 baseline | CIs present but verdict column not derived from the z-test, or baseline is the pretrained base instead of the SFT checkpoint | Z-tested verdict per row; baseline is the matched bf16 SFT checkpoint; `internal_task_v1` explicitly checked |
| **Throughput + memory** | Single tok/s number, no VRAM, no split | Throughput reported but prefill and decode lumped; VRAM reported at load time | Prefill and decode reported separately, median over warmed-up runs; steady-state VRAM; kernel path named per variant |
| **Recommendation + trade-off reasoning** | Picks a variant with no reasoning | Picks a variant with partial reasoning; build cost or runtime constraints ignored | Recommendation defended on accuracy, memory, decode speed, build time, and runtime-compatibility; runner-up and reject-case both argued |

**Pass threshold:** 7/10, with no single dimension at 0 and the "apples-to-apples setup" dimension at 2.

## 5. Common mistakes

- **Mixing base checkpoints.** Quantizing the pretrained base for one variant and the SFT model for another is the most common silent error; every row must come from the same starting weights.
- **Different group size across GPTQ and AWQ.** `g128` vs `g64` changes both the file size and the per-row quality. Keep the group size matched across the two offline variants, or report both configurations explicitly.
- **Calibrating on the wrong distribution.** Using wiki-text when the deployed traffic is code, or using a tiny 32-sample calibration set. 128-512 rows from the deployment distribution is the floor.
- **Comparing bnb at batch 1 to GPTQ/AWQ at batch 32** (or vice versa). bnb's slowdown is largest at high batch sizes where the dequant overhead dominates; keep the batch size identical across all four runs.
- **Claiming a "win" from a +0.5 MMLU point delta.** 500 items at p≈0.5 has a stderr of ~0.022; a 0.5-point delta is noise. Use the z-test; "flat" is a legitimate finding.
- **Reporting a single "tok/s" number.** Averaging prefill and decode over a short prompt gives one number and hides the fact that GPTQ/AWQ help decode disproportionately. Always split.
- **Using `desc_act=True` for GPTQ without noting the kernel cost.** The `desc_act` mode re-orders columns and triggers a slightly slower kernel in some vLLM versions; report both the accuracy and the throughput cost, and keep the same setting across every GPTQ comparison.
- **Picking AWQ `GEMV` for a vLLM deployment.** `GEMV` packing is batch-1-optimized and incompatible with vLLM's AWQ kernels. Always use `GEMM` for vLLM targets.
- **Shipping bitsandbytes for production serving on the strength of QLoRA familiarity.** QLoRA is training-time; bnb's inference kernels are not throughput-tuned. The exercise exists to make this distinction explicit.
- **Reporting on-disk file size as "memory savings" without steady-state VRAM.** The kv-cache dominates VRAM for long contexts; a 4-bit weight compression that cuts disk by 4× cuts VRAM by less, especially at longer sequence lengths.
- **Not re-running calibration when the checkpoint changes.** A new SFT run is a new checkpoint; its activation statistics are different. Treat the quantized artifact as downstream of the checkpoint and regenerate on every release.
- **Ignoring GSM8K drops.** Math is the first thing a bad quant kills. If `gsm8k` drops more than any other task, the quant is wrong for a reasoning-heavy workload — do not ship it just because MMLU looks fine.

## 6. References

- Frantar, Ashkboos, Hoefler, Alistarh — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. <https://arxiv.org/abs/2210.17323>
- Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han — *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. <https://arxiv.org/abs/2306.00978>
- Dettmers, Pagnoni, Holtzman, Zettlemoyer — *QLoRA: Efficient Finetuning of Quantized LLMs* (NF4, double-quant, and the bitsandbytes 4-bit path). <https://arxiv.org/abs/2305.14314>
- Hugging Face Transformers quantization documentation — `BitsAndBytesConfig`, GPTQ and AWQ integration via `optimum`. <https://huggingface.co/docs/transformers/main/en/quantization>
- vLLM documentation — supported quantization formats and kernel flags (`awq`, `awq_marlin`, `gptq`, `gptq_marlin`). <https://docs.vllm.ai/>
- bitsandbytes repository and quantization documentation. <https://github.com/bitsandbytes-foundation/bitsandbytes>
- AutoGPTQ repository (reference for `BaseQuantizeConfig`, `desc_act`, group-size semantics). <https://github.com/AutoGPTQ/AutoGPTQ>
- AutoAWQ repository (reference for `GEMM` vs `GEMV` packing). <https://github.com/casper-hansen/AutoAWQ>
- EleutherAI `lm-evaluation-harness` — same harness used in mod-108 for the accuracy rows. <https://github.com/EleutherAI/lm-evaluation-harness>
- NIST AI Risk Management Framework — *Manage 1.3* (release gates driven by measured performance) and *Measure 2.3* applies to the quant/accuracy/throughput trade-off table. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — *ML06 AI Supply Chain Vulnerabilities*: pinning `auto-gptq`, `autoawq`, and `bitsandbytes` versions and recording a sha256 of the calibration set is the supply-chain control for reproducible quantized artifacts. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — technique catalog; the per-sample `--log_samples` logs produced by the shared harness support regression-triage workflows described in the mitigations column. <https://atlas.mitre.org/>
