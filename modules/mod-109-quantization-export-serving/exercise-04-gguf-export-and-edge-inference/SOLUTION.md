# mod-109 · exercise-04 — GGUF Export and Edge Inference — Solution

## 1. Solution overview

The deliverable is an **end-to-end edge-inference artifact** for the fine-tuned checkpoint: a GGUF file produced by `llama.cpp`'s converter, re-quantized to three common widths (q8_0, q5_K_M, q4_K_M), evaluated for quality drift against the full-precision reference, benchmarked on CPU/laptop/consumer-GPU hardware, and shipped with a one-page recommendation matrix keyed to edge-device budgets.

Four ideas carry the exercise:

- **GGUF is a container, not a quantization scheme.** It is a single-file, mmap-friendly format that packs model weights, tokenizer, metadata, and a declared quantization layout together. The quant *scheme* (q4_K_M, q5_K_S, q8_0, ...) is independent of the container; the exercise converts to the container first with `convert_hf_to_gguf.py`, then re-quantizes with `llama-quantize`.
- **K-quants are the production default; the old Q4_0 is a reference point, not a target.** The *K*-quants (q2_K through q8_K in several sub-flavors `_S`, `_M`, `_L`) use per-block scales plus a super-block scale and deliver materially better quality per bit than the classic `q4_0` / `q5_0` layouts. Pick `q4_K_M` for a 4-bit baseline, `q5_K_M` for a quality-tilted mid-point, `q8_0` for a near-lossless reference.
- **Edge means the full matrix: CPU-only, Apple Silicon (Metal), small discrete GPUs, and mobile.** Throughput scales very differently by platform. A q4_K_M file that runs at 25 tok/s on an M3 Max may run at 6 tok/s on a Raspberry Pi 5 and at 70 tok/s on a 4060. Report one number per target platform, not a single "edge speed."
- **Quality drift must be measured against the same tokenizer and chat template.** `llama.cpp`'s chat-template detection has caveats: not every HF chat template is supported and silent fallback to a default template makes an instruct model look brain-damaged. Validate the template path explicitly with a handful of fixed prompts and compare the outputs byte-for-byte with the Transformers baseline at temperature 0.

The worked example takes `./runs/llama-3.2-3b-sft`, converts to GGUF, re-quantizes to q8_0 / q5_K_M / q4_K_M, measures perplexity on a held-out slice, measures generation tok/s on CPU-only (x86_64), M-series laptop, and a consumer GPU (RTX 4060 Ti 8 GB), and produces a decision table tied to deployment targets (desktop app, mobile app via Ollama, air-gapped kiosk).

## 2. Worked implementation

### 2.1 Environment and pins

```text
# Pin at a commit, not a tag; llama.cpp moves fast.
LLAMA_CPP_COMMIT=b3906              # confirm current release at run time
# Build flags that matter:
#   -DGGML_METAL=ON       (Apple Silicon)
#   -DGGML_CUDA=ON        (NVIDIA)
#   -DGGML_BLAS=ON        (CPU, OpenBLAS/MKL)
#   -DGGML_NATIVE=ON      (march=native; required for AVX2/AVX512 CPU perf)

# Python side (converter + eval)
python>=3.11
transformers==4.45.2
sentencepiece==0.2.0
gguf==0.10.0                        # llama.cpp's python package

# lm-eval reused from mod-108
lm-eval==0.4.5
```

Record the llama.cpp commit hash, the build flags, and the GGUF file sha256 in `env.txt` and in each GGUF's embedded `general.description` metadata.

### 2.2 Convert HF → GGUF (fp16 intermediate)

```bash
# convert.sh — HF safetensors to a single-file GGUF at bf16/fp16 fidelity
set -euo pipefail
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
git checkout $LLAMA_CPP_COMMIT
cmake -B build -DGGML_NATIVE=ON -DGGML_METAL=ON       # or -DGGML_CUDA=ON on Linux/NVIDIA
cmake --build build --config Release -j

python convert_hf_to_gguf.py \
  ../runs/llama-3.2-3b-sft \
  --outfile ../outputs/gguf/llama-3.2-3b-sft-f16.gguf \
  --outtype f16
```

Notes:

- `--outtype f16` keeps full weight precision inside the GGUF container; later quant stages read from this file. If the base is bf16, use `--outtype bf16` if the llama.cpp commit supports it; otherwise `--outtype f16` is numerically close.
- The converter writes the tokenizer and chat-template name into the GGUF metadata. Inspect with `./build/bin/llama-gguf-metadata ../outputs/gguf/llama-3.2-3b-sft-f16.gguf`. Confirm `tokenizer.chat_template` is present and matches the HF tokenizer's `chat_template`. If not, the next step bakes in a specific template with `--chat-template` at inference time, or the metadata is edited.

### 2.3 Re-quantize to three target widths

```bash
# quantize.sh — three target layouts from the single f16 intermediate
set -euo pipefail
cd llama.cpp/build/bin
for q in q8_0 q5_K_M q4_K_M; do
  ./llama-quantize ../../../outputs/gguf/llama-3.2-3b-sft-f16.gguf \
                    ../../../outputs/gguf/llama-3.2-3b-sft-${q}.gguf \
                    ${q}
done
```

Expected on-disk sizes for a 3.2B-parameter model:

| Quant | Approx size | Approx bits/param |
|---|---:|---:|
| f16          | ~6.4 GB | 16 |
| q8_0         | ~3.4 GB |  8.5 |
| q5_K_M       | ~2.3 GB |  5.7 |
| q4_K_M       | ~1.9 GB |  4.8 |

`q4_K_M` is the production 4-bit default. `q5_K_M` is the "a little more headroom at a modest size bump" option. `q8_0` is used as the near-lossless reference row.

### 2.4 Perplexity / reference comparison

Both for sanity-check and for a quick quality number. Use `llama.cpp`'s `llama-perplexity` on a short held-out slice (wiki.test or a task-specific slice), and lm-eval-harness for comparable multi-task numbers.

```bash
# perplexity.sh — wiki.test.raw is small; ~10-15 min on CPU for a 3B model at q4_K_M
set -euo pipefail
cd llama.cpp/build/bin
for q in f16 q8_0 q5_K_M q4_K_M; do
  ./llama-perplexity \
     -m ../../../outputs/gguf/llama-3.2-3b-sft-${q}.gguf \
     -f ../../wikitext-2-raw/wiki.test.raw \
     -c 2048 -b 512 \
     > ../../../results/perplexity_${q}.log
done
```

Record the final perplexity line per quant into `results/summary.csv`.

For harness-level comparability, run the lm-eval `gguf` runner against each file (requires a small `llama-server` wrapper):

```bash
# lm_eval_gguf.sh — run the mod-108 task set through the GGUF files via llama-server
set -euo pipefail
cd llama.cpp/build/bin
for q in f16 q8_0 q5_K_M q4_K_M; do
  ./llama-server -m ../../../outputs/gguf/llama-3.2-3b-sft-${q}.gguf \
                 -c 4096 --host 0.0.0.0 --port 8080 &
  PID=$!
  sleep 10                                        # wait for load
  lm_eval --model openai-chat-completions \
          --model_args base_url=http://localhost:8080/v1/chat/completions,model=local \
          --tasks mmlu,arc_challenge,hellaswag,gsm8k,internal_task_v1 \
          --num_fewshot 5 --seed 1234 --fewshot_random_seed 1234 \
          --output_path ../../../results/eval_${q}
  kill $PID; wait || true
done
```

### 2.5 Throughput on three edge platforms

```bash
# bench.sh — runs llama-bench against each file on the platform at hand
set -euo pipefail
cd llama.cpp/build/bin
PLATFORM=${1:-cpu}        # cpu | metal | cuda
case "$PLATFORM" in
  cpu)   NGL=0  ;;
  metal) NGL=99 ;;
  cuda)  NGL=99 ;;
esac
for q in f16 q8_0 q5_K_M q4_K_M; do
  ./llama-bench -m ../../../outputs/gguf/llama-3.2-3b-sft-${q}.gguf \
                -p 512 -n 128 -ngl $NGL -t $(nproc) -r 3 \
                > ../../../results/bench_${PLATFORM}_${q}.log
done
```

`llama-bench` outputs two numbers per model: *prompt-processing* tok/s (`pp512`) and *generation* tok/s (`tg128`). These correspond to prefill and decode and should be reported separately, just like exercise 01.

### 2.6 Sanity check: chat-template parity

A ten-prompt fixed set, temperature 0, compared Transformers (reference) vs. `llama-cli` (GGUF).

```python
# template_check.py — compares outputs on a fixed prompt set, reports divergences
import json, subprocess, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

PROMPTS = pathlib.Path("./data/template_check_prompts.jsonl").read_text().splitlines()
MODEL = "./runs/llama-3.2-3b-sft"
tok = AutoTokenizer.from_pretrained(MODEL)
ref = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
ref.eval()

divergences = []
for line in PROMPTS:
    row = json.loads(line)
    msgs = row["messages"]
    ref_input = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ref_ids = tok(ref_input, return_tensors="pt").to(ref.device)
    with torch.inference_mode():
        ref_out = ref.generate(**ref_ids, max_new_tokens=64, do_sample=False)
    ref_text = tok.decode(ref_out[0, ref_ids.input_ids.shape[1]:], skip_special_tokens=True)
    # llama-cli with --jinja applies the GGUF-embedded chat template
    gguf_text = subprocess.run(
        ["./llama.cpp/build/bin/llama-cli",
         "-m", "./outputs/gguf/llama-3.2-3b-sft-q8_0.gguf",
         "--jinja", "-p", json.dumps(msgs), "-n", "64", "--temp", "0", "--no-warmup", "-no-cnv"],
        capture_output=True, text=True, check=True).stdout.strip()
    if ref_text.strip() != gguf_text.strip():
        divergences.append({"prompt": msgs, "ref": ref_text, "gguf": gguf_text})
pathlib.Path("results/template_check.json").write_text(json.dumps(divergences, indent=2))
print(f"{len(divergences)}/{len(PROMPTS)} divergent; see results/template_check.json")
```

`q8_0` is the right file to run this against — any divergence at q8_0 is a template/tokenizer issue, not a quant issue. Investigate before moving to q5_K_M or q4_K_M.

### 2.7 Worked results table (illustrative; numbers depend on hardware)

Reference numbers from the pipeline above on three platforms:

- **CPU-only**: 12-core x86_64 (AVX2+AVX512), 32 GB DDR4, no GPU
- **Laptop (Metal)**: Apple M3 Max, 36 GB unified memory
- **Consumer GPU**: RTX 4060 Ti 8 GB + Ryzen 5 7600

| Quant   | Size GB | PPL (wiki) | MMLU | GSM8K | CPU pp512 / tg128 | Metal pp512 / tg128 | 4060Ti pp512 / tg128 |
|---------|--------:|-----------:|-----:|------:|--------------------:|---------------------:|----------------------:|
| f16     | 6.4     | 9.42       | 0.526 | 0.258 | 180 / 3.2  | 820 / 18 | 2100 / 54 |
| q8_0    | 3.4     | 9.44       | 0.525 | 0.256 | 240 / 5.8  | 1100 / 32 | 2500 / 62 |
| q5_K_M  | 2.3     | 9.52       | 0.521 | 0.253 | 290 / 7.4  | 1300 / 42 | 2700 / 70 |
| q4_K_M  | 1.9     | 9.71       | 0.515 | 0.247 | 320 / 8.9  | 1450 / 49 | 2900 / 78 |

Pattern to read:

- **Perplexity grows monotonically as bits/param drop.** The q8_0 vs. f16 delta is within rounding (0.02 PPL on 3B); q5_K_M adds ~0.1 PPL, q4_K_M another ~0.2. These are small for most tasks; large for math (GSM8K drops the most).
- **Generation tok/s grows when bits-per-param drops** because decode is memory-bandwidth-bound; a smaller model fits more aggressively in L2/L3 cache (CPU) or in on-chip memory (M-series / 4060).
- **Prefill scales similarly** because the dequant kernel is fused with matmul in llama.cpp's `mul_mat_q` path, so smaller weights still save bandwidth on the prefill phase.
- **The edge target decides the row.** On a 16 GB laptop, `q5_K_M` is the sweet spot (23 tok/s or faster, 2.3 GB file). On a Raspberry Pi 5 (not in the table; expect single-digit tok/s), `q4_K_M` is the only one that is both fast and small enough. On the 4060 Ti, `q4_K_M` is still fastest but the quality cost may be noticeable on math tasks — `q5_K_M` is the safer default if PPL matters.

### 2.8 Directory layout (as delivered)

```
outputs/exercise-04/
├── env.txt                          # llama.cpp commit, build flags, GPU/CPU info
├── convert.sh
├── quantize.sh
├── perplexity.sh
├── lm_eval_gguf.sh
├── bench.sh
├── template_check.py
├── data/
│   ├── template_check_prompts.jsonl
│   └── wiki.test.raw                # downloaded offline
├── outputs/gguf/
│   ├── llama-3.2-3b-sft-f16.gguf
│   ├── llama-3.2-3b-sft-q8_0.gguf
│   ├── llama-3.2-3b-sft-q5_K_M.gguf
│   └── llama-3.2-3b-sft-q4_K_M.gguf
├── results/
│   ├── perplexity_*.log
│   ├── eval_*/ (lm-eval outputs)
│   ├── bench_{cpu,metal,cuda}_*.log
│   ├── template_check.json
│   ├── summary.csv
│   └── gguf_sha256.txt
└── report.md
```

### 2.9 `report.md` skeleton

```markdown
# GGUF export + edge inference — report

## Setup
- Source checkpoint: ./runs/llama-3.2-3b-sft (bf16 safetensors)
- llama.cpp commit: b3906 (sha256 of build artifact in env.txt)
- Build flags: metal on (laptop), cuda on (desktop), blas on (cpu)
- Platforms:
    - CPU: 12-core x86_64 w/ AVX2+AVX512, 32 GB DDR4
    - Metal: Apple M3 Max, 36 GB unified
    - CUDA: RTX 4060 Ti 8 GB + Ryzen 5 7600
- Eval: wiki.test.raw PPL + mod-108 task set via llama-server + lm-eval
- Chat-template: HF template embedded in GGUF; 10-prompt parity check at q8_0

## Results
(see table in §2.7 and results/summary.csv)

## Decision
For the three target edge profiles:
- Desktop/laptop with 16 GB+: **q5_K_M**. Within 1 point MMLU of f16, 2.3 GB on disk, >20 tok/s on M3 Max.
- Low-end (Raspberry Pi 5 / mid-tier laptop): **q4_K_M**. 1.9 GB on disk, 8-10 tok/s on CPU; the 1-point MMLU and 1-point GSM8K regression is the cost.
- Near-lossless archival / reference row: **q8_0**. 3.4 GB; runs everywhere llama.cpp runs; used for template parity and as the quality-anchor reference row in release notes.

Reject `q2_K` and `q3_K_*` for this checkpoint — the quality cost on reasoning
tasks (GSM8K, internal_task_v1) is disproportionate.

## Risks / follow-ups
1. Chat-template drift: re-run template_check.py on every quant file after a
   llama.cpp bump; template parsing has regressed between versions in the past.
2. Mobile deployment (iOS/Android via llama.cpp or Ollama): test on-device
   thermals — sustained tok/s drops 30-50% after 2-3 minutes.
3. The downstream Ollama model definition (Modelfile) should pin the GGUF
   sha256 so model-registry pulls are reproducible.
```

## 3. Validation steps

1. **GGUF metadata is complete.** `llama-gguf-metadata` on each file shows `general.architecture`, `tokenizer.chat_template`, `tokenizer.bos/eos_token_id`, and a non-empty `general.description` with the build info. Missing metadata means `llama-cli` will guess at runtime.
2. **Chat-template parity checked at q8_0.** `template_check.json` is zero-length or divergences are investigated and explained. A silent template fallback is the most common "my GGUF is broken" bug.
3. **Perplexity is monotonic across widths.** f16 ≤ q8_0 ≤ q5_K_M ≤ q4_K_M, with reasonable step sizes (≤ 0.3 PPL between adjacent widths on this corpus). A q4_K_M file that is *lower* PPL than q8_0 is a measurement error; re-run with the same context length.
4. **lm-eval numbers reused from mod-108.** Same task YAML, same seeds, same chat-template decision. The quant rows are comparable to the bf16 rows; the chat template path is the only moving piece.
5. **Throughput is split into pp and tg.** `llama-bench` reports both; a report that only gives one number has not followed the harness.
6. **Hardware is named on every row.** "25 tok/s" is not a specification; "25 tok/s tg128 on M3 Max 36 GB, ngl=99" is.
7. **Build flags recorded.** `-DGGML_METAL=ON` vs. CPU-only is a 5-10× delta on Apple Silicon; the flag used for each platform row is in `env.txt`.
8. **File sha256 recorded.** `gguf_sha256.txt` captures the sha256 of every produced `.gguf`; the Modelfile or Ollama pin references these.
9. **A template check prompt set exists and is stored.** `data/template_check_prompts.jsonl` is small but *versioned* in the repo; the parity check is not a one-off.
10. **CUDA, Metal, and CPU runs are all present.** The exercise is edge-targeted; a report that measures only CUDA has not delivered on the edge brief.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Conversion pipeline** | HF checkpoint not converted, or converted without tokenizer / chat-template metadata | Converted but only one quant width produced, or the f16 intermediate is skipped | f16 intermediate + three K-quants (q8_0, q5_K_M, q4_K_M) produced with metadata and sha256 |
| **Chat-template parity** | No parity check; GGUF outputs diverge silently from Transformers reference | Parity check run only at q4_K_M (where a divergence could be quant OR template) | Parity check at q8_0, divergence count recorded, any divergence investigated |
| **Quality evaluation** | No perplexity, no task eval | PPL only, or lm-eval run at inconsistent seeds vs. mod-108 | PPL on wiki.test and lm-eval task set with mod-108 seeds; monotonic PPL confirmed |
| **Throughput coverage** | Single platform, single number | Two platforms, or prefill/decode lumped | At least three platforms (CPU + Metal + CUDA), prefill and decode reported separately, hardware fully named |
| **Decision matrix** | Picks one quant with no device link | Picks one per device without reasoning | Explicit matrix mapping edge profile → quant choice, with PPL/MMLU/GSM8K cost named per choice |

**Pass threshold:** 7/10, with no dimension at 0 and chat-template parity at 2.

## 5. Common mistakes

- **Converting with `--outtype q4_K_M` directly.** The converter's quant path has fewer knobs than `llama-quantize`; always produce an f16 intermediate first, then quantize from it.
- **Not inspecting GGUF metadata.** A GGUF with no `tokenizer.chat_template` falls back to `llama.cpp`'s default template, which does not match the HF chat-tuned model. The output looks wrong even though the weights are fine.
- **Running `llama-perplexity` with different context lengths across quant widths.** `-c 2048` on one, `-c 4096` on another — the PPL numbers are not comparable. Pin the context length.
- **Benchmarking without `-DGGML_NATIVE=ON` on CPU.** `march=native` is what enables AVX2/AVX512 in the quant kernels; a non-native build can be 2-4× slower for no good reason.
- **Reporting `metal` benchmarks without `-ngl 99`.** Without the layer-offload flag, llama.cpp runs the model on CPU and Metal is unused.
- **Treating Q4_0 as a sensible 4-bit target.** The legacy `q4_0` layout is worse than `q4_K_M` at the same size budget; do not include it in the production matrix without an explicit reason.
- **Skipping the lossless reference row.** Without q8_0 (or f16), you cannot distinguish quant-induced regressions from checkpoint-level regressions.
- **Running the perplexity measurement on a corpus from the training distribution.** The perplexity number is meaningful only on a held-out corpus; wiki.test is the standard reference because it is not in most SFT sets, but confirm for your checkpoint.
- **Deploying without pinning the llama.cpp commit.** The GGUF format is stable but the Jinja chat-template engine and the quant kernels are not; a `llama.cpp` bump can regress template handling or decode speed. Pin the commit in the Modelfile.
- **Reading Metal throughput on a laptop under battery.** Apple Silicon throttles under battery; benchmark on AC power and name the condition.
- **Ignoring thermal throttling on the Pi / phone.** Tok/s measured over 10 seconds on a cold device is an overestimate of sustained throughput. Report both a 10-second burst and a 3-minute sustained number for edge platforms.
- **Shipping the f16 file as "edge."** 6 GB is not an edge artifact for most devices; it is included only as the quality anchor.

## 6. References

- `llama.cpp` repository — `convert_hf_to_gguf.py`, `llama-quantize`, `llama-bench`, `llama-perplexity`, `llama-server`. <https://github.com/ggerganov/llama.cpp>
- GGUF format specification (official, in-tree). <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
- `llama.cpp` quantization documentation (K-quants layout, per-block scales). <https://github.com/ggerganov/llama.cpp/blob/master/docs/quantize.md>
- Hugging Face Transformers — tokenizer chat-template semantics; the template that ships in the GGUF comes from `tokenizer.chat_template`. <https://huggingface.co/docs/transformers/chat_templating>
- EleutherAI `lm-evaluation-harness` — `openai-chat-completions` runner used against `llama-server`. <https://github.com/EleutherAI/lm-evaluation-harness>
- Ollama Modelfile reference — downstream deployment surface for GGUF files; pin `FROM` to a sha256. <https://github.com/ollama/ollama/blob/main/docs/modelfile.md>
- NIST AI Risk Management Framework — *Manage 2.1* (performance monitoring across deployment environments) applies directly to the per-platform throughput matrix. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — *ML06 AI Supply Chain Vulnerabilities*: pinning the `llama.cpp` commit, recording the GGUF sha256, and embedding `general.description` is the supply-chain control for edge artifacts. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — technique catalog; edge-deployed models are exposed to *Access ML Model* and *Reconstruct ML Model* techniques (device compromise, model-file extraction) that should be covered in the risk section. <https://atlas.mitre.org/>
