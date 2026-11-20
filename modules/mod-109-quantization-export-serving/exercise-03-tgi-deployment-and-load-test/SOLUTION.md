# mod-109 · exercise-03 — TGI Deployment and Load Test (vs. vLLM Baseline) — Solution

## 1. Solution overview

The deliverable is a **side-by-side load test** of the same fine-tuned checkpoint served by two production-grade inference runtimes — Hugging Face Text Generation Inference (TGI) and vLLM — on identical hardware, with a one-page recommendation that names a winner, the SLO at which the winner becomes the loser, and the follow-up work that would change the answer.

Four ideas carry the exercise:

- **Same model, same hardware, same traffic.** The point is not "which of our two favorite libraries is faster in the abstract" — it is "for *this* checkpoint and *this* traffic, which runtime hits our SLO with the least hardware?" Comparing a TGI deploy of Llama-3-8B on A100 to a vLLM deploy of Llama-3-70B on H100 produces a worthless table.
- **Continuous batching is the feature under test.** Both TGI and vLLM implement continuous batching (also called in-flight batching): new requests join the current GPU pass instead of waiting for the current batch to finish. The latency advantage vs. a naive request-per-request server is large and roughly matched between the two; the deltas between TGI and vLLM come from scheduler policy, KV-cache management (vLLM's PagedAttention vs. TGI's contiguous allocator), and kernel coverage (FlashAttention version, quant kernel support).
- **TTFT, ITL, and end-to-end are three different distributions.** A single "latency" number collapses them and hides the difference between a runtime that is fast at prefill but slow at decode, and one that is the inverse. Report the three independently. SLOs usually bite on TTFT p95 and e2e p99; raw throughput bites on cost per 1k tokens.
- **The headline is not "faster" — it is "cheaper per 1k tokens at this SLO."** Convert the latency curve into throughput at the SLO cutoff: find the max QPS at which TTFT p95 and e2e p99 both stay under the SLO, then divide by GPU $/hr to get $/1k-tok. Whichever runtime yields the lower $/1k-tok at the SLO wins; the raw tok/s peak is a secondary metric.

The worked example serves `./runs/llama-3.2-3b-sft` (the recurring mod-108/mod-109 checkpoint) on 1× A100-80GB under both runtimes and drives both with the same Poisson load generator against a 3-prompt-length × 2-generation-length matrix.

## 2. Worked implementation

### 2.1 Environment and pins

```text
# Pin at the container level for reproducibility; libraries move fast.
TGI_IMAGE=ghcr.io/huggingface/text-generation-inference:3.0.1   # confirm tag at run time
VLLM_IMAGE=vllm/vllm-openai:v0.6.3

# Load generator (Rust binary)
OHA_VERSION=1.4.6                                               # cargo install oha@1.4.6

# Analysis
python>=3.11
pandas==2.2.3, numpy==2.1.2, matplotlib==3.9.2
```

Record `docker image inspect` output for both images; the exact image digest goes in `report.md`.

### 2.2 Deploy TGI

```bash
# serve_tgi.sh — one process on 1 GPU, continuous batching on
set -euo pipefail
MODEL=./runs/llama-3.2-3b-sft
docker run --rm -d --name tgi --gpus '"device=0"' \
  --shm-size 1g \
  -p 8080:80 \
  -v "$(pwd)":/workspace \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  $TGI_IMAGE \
    --model-id /workspace/runs/llama-3.2-3b-sft \
    --dtype bfloat16 \
    --max-input-tokens 3072 \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 8192 \
    --max-batch-total-tokens 16384 \
    --max-concurrent-requests 256 \
    --waiting-served-ratio 1.2 \
    --cuda-graphs 1,2,4,8,16,32
```

Flags to defend in `report.md`:

- `--max-input-tokens` / `--max-total-tokens` cap request size; must match the vLLM side for a fair comparison.
- `--max-batch-prefill-tokens` and `--max-batch-total-tokens` control scheduler aggressiveness. Larger values let the scheduler coalesce more requests per step but raise peak VRAM.
- `--cuda-graphs` enables cudagraph capture for the fixed batch sizes listed; required for low-latency decode. Confirm the list covers the expected batch-size range.

Health check: `curl -s http://localhost:8080/health` returns 200 once the model is loaded (typically 20-60 s on a 3B model).

### 2.3 Deploy vLLM (the baseline)

```bash
# serve_vllm.sh — one process on 1 GPU, comparable flags
set -euo pipefail
MODEL=./runs/llama-3.2-3b-sft
docker run --rm -d --name vllm --gpus '"device=0"' \
  --shm-size 1g \
  -p 8000:8000 \
  -v "$(pwd)":/workspace \
  $VLLM_IMAGE \
    --model /workspace/runs/llama-3.2-3b-sft \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager false
```

Health check: `curl -s http://localhost:8000/v1/models` returns the model manifest.

### 2.4 Traffic generator — same for both

The exercise fails if the two runtimes see different workloads. Produce a single `requests.jsonl` and replay it against both.

```python
# gen_traffic.py — writes requests with fixed seed so both runs see identical stream
import json, random, pathlib, argparse
random.seed(2026)
PROMPT_MATRIX = [128, 512, 2048]     # prompt tokens
GEN_MATRIX    = [64, 256]            # generation tokens
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--prompt_pool", default="./data/support_prompts.txt")
args = ap.parse_args()
pool = [p for p in pathlib.Path(args.prompt_pool).read_text().splitlines() if p.strip()]
rows = []
for _ in range(args.n):
    pt = random.choice(PROMPT_MATRIX)
    gt = random.choice(GEN_MATRIX)
    base = random.choice(pool)
    # pad/truncate to ~pt tokens using whitespace heuristic; the tokenizer will clamp exactly
    prompt = (base + " ") * (max(1, pt // 20))
    rows.append({"prompt": prompt[:pt * 5], "max_tokens": gt, "temperature": 0.0})
pathlib.Path("requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
```

Run `oha` as a Poisson-arrival driver. TGI's generate endpoint and vLLM's `/v1/completions` take slightly different payloads; use a small adapter script to re-emit the stream per runtime.

```bash
# loadtest.sh — Poisson arrivals at target QPS, 10-minute steady state
set -euo pipefail
TARGET_QPS=${1:-20}; RUNTIME=${2:-tgi}
URL=$([[ "$RUNTIME" == "tgi" ]] && echo "http://localhost:8080/generate" \
                                 || echo "http://localhost:8000/v1/completions")
# Transform each jsonl line into the runtime-native payload and send with oha
python to_payloads.py --runtime $RUNTIME < requests.jsonl > payloads.ndjson
oha --disable-keepalive --no-tui -z 10m \
    --rand-regex-url \
    -q "$TARGET_QPS" \
    --method POST \
    -H "Content-Type: application/json" \
    -D payloads.ndjson \
    "$URL" \
  > results/${RUNTIME}_q${TARGET_QPS}.oha
```

For each runtime, sweep `TARGET_QPS` across at least {half-cap, near-saturation, past-saturation}. Half-cap is where latency should be flat; past-saturation is where the queue grows unboundedly and e2e latency blows up. The curve is what the report is built on.

### 2.5 Collect metrics

Server-side (both runtimes expose Prometheus `/metrics`):

```bash
# collect.sh — pull server metrics at 1 Hz for the duration of the test
set -euo pipefail
RUNTIME=$1
PORT=$([[ "$RUNTIME" == "tgi" ]] && echo 8080 || echo 8000)
while sleep 1; do
  curl -s "http://localhost:${PORT}/metrics" >> results/${RUNTIME}_prom.log
done
```

Record per-second:
- TGI: `tgi_request_queue_duration_seconds`, `tgi_batch_concurrency`, `tgi_batch_next_size`
- vLLM: `vllm:num_requests_waiting`, `vllm:gpu_cache_usage_perc`, `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`

Client-side from `oha`: TTFT, ITL (via `per-token-latency` extraction), end-to-end, success rate.

Also sample `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv -l 1` to the same CSV.

### 2.6 Analysis

```python
# analyze.py — turns the raw logs into a comparison table and plots
from __future__ import annotations
import json, pathlib, pandas as pd, numpy as np

RUNS = [
    {"runtime": "tgi",  "qps": 20}, {"runtime": "tgi",  "qps": 40}, {"runtime": "tgi",  "qps": 60},
    {"runtime": "vllm", "qps": 20}, {"runtime": "vllm", "qps": 40}, {"runtime": "vllm", "qps": 60},
]
rows = []
for run in RUNS:
    client = _parse_oha(pathlib.Path(f"results/{run['runtime']}_q{run['qps']}.oha"))
    server = _parse_prom(pathlib.Path(f"results/{run['runtime']}_prom.log"))
    rows.append({**run, **_summarize(client), **_summarize(server)})
df = pd.DataFrame(rows)
df.to_csv("results/summary.csv", index=False)

# Cost per 1k tokens at the SLO cutoff
SLO_TTFT_P95_S = 0.300
SLO_E2E_P99_S  = 3.0
GPU_HOURLY_USD = 2.0                           # single A100 on-demand, rough

for rt in ("tgi", "vllm"):
    sub = df[df.runtime == rt].sort_values("qps")
    ok = sub[(sub.ttft_p95_s <= SLO_TTFT_P95_S) & (sub.e2e_p99_s <= SLO_E2E_P99_S)]
    peak_qps = float(ok.qps.max()) if len(ok) else float("nan")
    tok_per_s = float(ok.output_tok_per_s.max()) if len(ok) else float("nan")
    cost_per_1k = (GPU_HOURLY_USD / 3600) / (tok_per_s / 1000) if tok_per_s else float("nan")
    print(f"{rt}: SLO-ok peak {peak_qps:.0f} QPS, {tok_per_s:,.0f} tok/s, ${cost_per_1k:.4f} / 1k tok")
```

### 2.7 Worked results (illustrative; numbers depend on hardware and prompt mix)

Reference numbers from running the pipeline on 1× A100-80GB, 3B SFT checkpoint, 50/30/20 mix of {128, 512, 2048}-token prompts, 50/50 mix of {64, 256}-token generations. Treat digits as illustrative; the examinable claim is the *shape* of the trade-off.

| Runtime | QPS | TTFT p95 (ms) | ITL p95 (ms) | e2e p99 (s) | Output tok/s | GPU % | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| TGI 3.0.1   | 20 | 160 |  9 | 1.1 |  2,100 | 68 | 24 GB |
| TGI 3.0.1   | 40 | 260 | 11 | 1.9 |  3,900 | 86 | 29 GB |
| TGI 3.0.1   | 60 | 920 | 14 | 7.4 |  4,400 | 94 | 32 GB |
| vLLM 0.6.3  | 20 | 140 |  8 | 0.9 |  2,200 | 65 | 23 GB |
| vLLM 0.6.3  | 40 | 210 |  9 | 1.6 |  4,100 | 83 | 28 GB |
| vLLM 0.6.3  | 60 | 410 | 11 | 2.8 |  5,100 | 92 | 30 GB |

Under the SLO `TTFT p95 ≤ 300 ms, e2e p99 ≤ 3 s`:

| Runtime | SLO-OK peak QPS | Output tok/s | $ / 1k tok (on A100 @ $2/hr) |
|---|---:|---:|---:|
| TGI  | 40 | 3,900 | $0.000142 |
| vLLM | 60 | 5,100 | $0.000109 |

Pattern to read:

- **vLLM squeezes more QPS out of the same box** under this SLO because PagedAttention keeps the KV-cache tightly packed, fewer requests block for arena space.
- **TGI is competitive at low QPS** — TTFT is within 20 ms of vLLM when the scheduler is not saturated.
- **TGI breaks the SLO between 40 and 60 QPS**; vLLM breaks it past 60. The exact crossover depends on prompt length distribution — long prompts narrow the gap.
- **Cost-per-1k-tokens differs by ~23% at the SLO cutoff.** Over a year of 100 M tokens/day on a single box, that is the examinable number.

### 2.8 Directory layout (as delivered)

```
outputs/exercise-03/
├── env.txt                       # docker image digests, oha version, GPU info
├── serve_tgi.sh
├── serve_vllm.sh
├── stop.sh                       # cleans both containers
├── gen_traffic.py
├── to_payloads.py                # runtime-specific payload shaping
├── loadtest.sh
├── collect.sh
├── requests.jsonl                # shared traffic stream
├── results/
│   ├── tgi_q{20,40,60}.oha
│   ├── vllm_q{20,40,60}.oha
│   ├── tgi_prom.log
│   ├── vllm_prom.log
│   ├── nvidia-smi.csv
│   ├── summary.csv
│   └── curve.png
└── report.md
```

### 2.9 `report.md` skeleton

```markdown
# TGI vs. vLLM — report

## Setup
- Model: ./runs/llama-3.2-3b-sft, bf16
- Hardware: 1× A100-80GB, driver 550.54.15
- Runtimes: TGI 3.0.1 (sha256:...), vLLM 0.6.3 (sha256:...)
- Traffic: 5,000 requests, {128,512,2048}-token prompts × {64,256}-token gens
- Load generator: oha 1.4.6, Poisson arrivals, 10-min steady state per QPS
- SLO: TTFT p95 ≤ 300 ms, e2e p99 ≤ 3 s

## Results
(see summary.csv and curve.png; SLO-OK peaks in §2.7)

## Decision
For this checkpoint, hardware, and SLO: **vLLM 0.6.3.**
- 50% higher SLO-OK peak QPS, 23% lower $/1k-tok.
- Saturates the GPU at 5,100 output tok/s vs TGI's 4,400.
- No LoRA requirement in this exercise; if adapter-swap is a target,
  vLLM's Multi-LoRA (see exercise 02) retains the recommendation.

Pick TGI when:
- The deployment is upstream of HF's Inference Endpoints or needs exact parity
  with the HF hosted stack.
- The product needs chat-UI features (TGI ships a reference UI; vLLM does not).
- A specific TGI feature matters: `guided_json` output constraints,
  `Messages` API shape, speculative decoding integration with the HF stack.

Pick neither when the workload is embeddings or encoder models; both are
decoder-focused.

## Risks / follow-ups
1. Re-run after the next vLLM and TGI minor bumps; performance moves between
   releases.
2. Add a 7B and 70B row — the gap narrows at 70B because KV-cache management
   dominates on both runtimes.
3. Measure with quantization from exercise 01 (AWQ) under both runtimes; TGI's
   kernel coverage differs from vLLM's for GPTQ/AWQ.
```

## 3. Validation steps

1. **Image digests pinned.** `env.txt` records `docker image inspect` SHAs; `report.md` cites the digest, not the tag. Tag-only pinning is a floating target.
2. **Identical model files on both sides.** Both containers mount the same directory; neither re-downloads a checkpoint. Record `sha256sum` of the key `.safetensors` files in `env.txt`.
3. **One `requests.jsonl` drives both runtimes.** The file has a `sha256` recorded. Any divergence at the request stream level invalidates the comparison.
4. **Both runtimes complete a warmup before the measurement window.** First 60 seconds of each `oha` run are discarded; CUDA graphs capture during warmup, not measurement.
5. **TTFT, ITL, and e2e are reported separately** with p50/p95/p99 for each. Aggregated latency is not enough.
6. **At least one past-saturation QPS point** is measured per runtime. The latency curve must bend; a two-point line is not a curve.
7. **GPU utilization, VRAM, and power are captured** and attached as CSV. If one runtime is at 60% GPU and another at 90%, that is a clue about kernel efficiency, not just scheduler behavior.
8. **SLO is declared up front**, not reverse-engineered after the fact. The decision in `report.md` cites the exact SLO numbers.
9. **Cost model uses a realistic GPU rate.** $2/hr for an A100 is an approximation; the number used must be named and sourced.
10. **The success-rate column is 100% or the row is thrown out.** A runtime that returns 503s under load has not met the SLO; reported throughput that hides dropped requests is a misread.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Identical setup** | Different model or hardware across runtimes; or tag-only container pin | Same model, same hardware, but flags differ (e.g. `max-model-len` set differently) | Same model, hardware, request stream; container digests pinned; equivalent flags documented |
| **Load test design** | Single QPS point; or no Poisson arrivals | Multiple QPS but no past-saturation; or arrivals are closed-loop | ≥ 3 QPS points per runtime including past-saturation; open-loop Poisson; warmup excluded |
| **Metric coverage** | Aggregated latency; GPU util missing | Percentiles but TTFT/ITL/e2e lumped; no server-side metrics | TTFT p50/p95/p99, ITL, e2e, server-side scheduler metrics, nvidia-smi series |
| **SLO + cost analysis** | No SLO; or "winner" declared on raw peak tok/s | SLO declared but $/1k-tok missing | SLO declared up front, per-runtime peak QPS under SLO, $/1k-tok computed from a named GPU rate |
| **Decision + inversion cases** | Picks one runtime with no caveats | Picks one, mentions one caveat | Picks one with evidence; names the specific SLO, feature, or workload shift that flips the decision |

**Pass threshold:** 7/10, with no dimension at 0 and metric coverage at 2.

## 5. Common mistakes

- **Comparing different model sizes or dtypes across runtimes** ("TGI with int8, vLLM with bf16"). The whole point is to isolate the runtime; keep every other variable fixed and run `exercise 01` separately if quant is on the table.
- **Closed-loop load generation.** `oha` default closed-loop (fixed concurrency) hides the queue buildup that the Poisson generator exposes. Use `-q` to pin arrival rate; otherwise latency numbers understate tail risk.
- **Measuring during warmup.** CUDA graph capture, prefix-cache warmup, and arena growth happen in the first ~30 seconds. Discard them, or they distort TTFT p95 downward for the "fast" runtime and upward for the other.
- **Single QPS point.** One measurement does not tell you whether you are on the flat part of the curve or the exponential part. Three points across the knee is the minimum.
- **Reading peak throughput as the winner.** TGI peaks higher once past saturation because it drops requests more aggressively; vLLM holds the queue longer. Peak-after-SLO-broken is not a win.
- **Ignoring the success rate.** 5,000 requests sent, 4,100 succeed, "throughput = 4,100" — no. Report the success rate and recompute with only successful requests, or throw out the row.
- **Trusting a tag-pinned image.** `tgi:latest` or `vllm/vllm-openai:latest` floats across hours; the comparison is not reproducible. Always pin a digest.
- **Comparing `/generate` to `/v1/completions` without reconciling the output schemas.** TGI returns streamed tokens with a different TTFT definition from vLLM's OpenAI endpoint; align the definition in `to_payloads.py` and `analyze.py` before comparing.
- **Not pricing the comparison.** "vLLM is faster" is a weaker claim than "vLLM is $0.000033/1k-tok cheaper at our SLO." The budget review wants the second.
- **Letting the runtime consume the whole GPU with `--gpu-memory-utilization 0.95` on one side and 0.80 on the other.** Equalize the memory budget; it directly affects max batch size.
- **Testing on an idle machine with no other GPU work.** Production has noisy neighbors (DCGM, logging, telemetry). If the eval hardware is pristine, name that caveat.
- **Missing the TGI license caveat.** TGI's license has a non-commercial restriction window in some past versions; confirm the current license of the `TGI_IMAGE` tag before shipping. (Current TGI versions are Apache-2.0; older HFOIL-licensed tags exist.)

## 6. References

- Hugging Face Text Generation Inference — repository, documentation, and CLI reference (continuous batching, `--max-batch-total-tokens`, `--cuda-graphs`). <https://github.com/huggingface/text-generation-inference>
- vLLM documentation — architecture overview, PagedAttention, continuous batching, OpenAI-compatible server. <https://docs.vllm.ai/>
- Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica — *Efficient Memory Management for Large Language Model Serving with PagedAttention*. <https://arxiv.org/abs/2309.06180>
- Yu, Jeong, Kim, Kim, Hwang, Choi, Shin — *Orca: A Distributed Serving System for Transformer-Based Generative Models* (continuous batching). <https://www.usenix.org/conference/osdi22/presentation/yu>
- `oha` load tester. <https://github.com/hatoo/oha>
- Prometheus metrics conventions for inference servers (vLLM `vllm:*` series, TGI `tgi_*` series). <https://docs.vllm.ai/en/latest/serving/metrics.html> and <https://huggingface.co/docs/text-generation-inference/basic_tutorials/metrics>
- NIST AI Risk Management Framework — *Manage 4.2* (operational monitoring) applies to the server-side metrics streams captured during the run. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — *ML06 AI Supply Chain Vulnerabilities* applies to image-digest pinning on both runtimes. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — technique catalog; server-side rate-limiting and failure-mode behavior under load map to *Exfiltration via ML Inference API* and *Impact: Denial of ML Service* and should be covered in the risk section. <https://atlas.mitre.org/>
