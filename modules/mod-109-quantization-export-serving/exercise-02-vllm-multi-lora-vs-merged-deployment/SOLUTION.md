# mod-109 · exercise-02 — vLLM Multi-LoRA vs. Merged Deployment — Solution

## 1. Solution overview

The deliverable is a **head-to-head benchmark and cost model** that picks between two ways of serving several LoRA adapters that share a common base model:

- **Multi-LoRA** — a single vLLM server loads the base model once and hot-swaps LoRA adapters at request time, routing each request to the adapter named in its payload. Enabled by `--enable-lora` plus `--lora-modules name=path ...` and selected per request via the `"model": "<adapter-name>"` field in the OpenAI-compatible API.
- **Merged** — each adapter is fused into a copy of the base model with `PeftModel.merge_and_unload()` and served as its own full checkpoint, either as N vLLM replicas (one per adapter) or as N tenants behind a router.

Four ideas carry the exercise:

- **Per-adapter traffic shape is the whole decision.** When traffic is spread across many low-QPS adapters, Multi-LoRA shares the base weights and the KV cache arena across all of them — one GPU serves them all. When one adapter dominates and needs peak throughput, that adapter is better served *merged* on dedicated hardware; the LoRA runtime overhead is pure deadweight.
- **Multi-LoRA has a measurable per-token overhead, and its bound is `max_loras`.** vLLM's PunicaKernel (or TritonLora backend) applies the `A·B` low-rank delta inside the matmul loop. The extra memory traffic shows up as a few percent of decode latency vs. the merged baseline. Peak concurrency per adapter is capped by `--max-loras` (how many are *simultaneously resident* on the GPU) and `--max-cpu-loras` (how many are kept hot in system memory). Request-level `max_loras_per_step` governs how many *distinct* adapters can appear in one batch.
- **Merge+unload is lossless but it is a *deployment* decision, not a *training* decision.** `merge_and_unload()` folds A·B back into the base weight matrix to byte-exact precision in fp32 (within rounding in bf16). It removes the ability to A/B test adapters cheaply because the base copy is duplicated per tenant.
- **The cost model is not latency; it's GPU-hours per request per tenant.** Multi-LoRA shares base-weight VRAM across N tenants; merged pays the base-weight footprint N times. For 4–8 small adapters on a 70B base, that gap is the exercise's main finding.

The worked example serves three adapters over a 7B base (`meta-llama/Llama-3.1-8B-Instruct` acting as the common base, with LoRA ranks 16/32/64 across three customer-support variants), compares a single Multi-LoRA vLLM instance to three merged vLLM instances, and produces a decision table keyed on traffic mix and SLO.

## 2. Worked implementation

### 2.1 Environment and pins

```text
# requirements.txt — match the mod-109 stack
torch==2.4.1
transformers==4.45.2
peft==0.13.2
accelerate==1.0.1
vllm==0.6.3
# load-test harness
oha==1.4.6         # installed via `cargo install oha` or binary release; not pip
```

### 2.2 Inputs: a shared base and three LoRA adapters

The three adapters come from prior exercises or are stand-ins for distinct task tunes:

```text
adapters/
├── support-general/     # r=16, alpha=32, dropout=0.0
├── support-billing/     # r=32, alpha=64
└── support-escalation/  # r=64, alpha=128
```

Every adapter's `adapter_config.json` references the same base `meta-llama/Llama-3.1-8B-Instruct`. If any adapter targets different modules or has a different `base_model_name_or_path`, Multi-LoRA will refuse to co-load it — fix the training run before running the exercise.

### 2.3 Variant A — Multi-LoRA serve

```bash
# serve_multi.sh — one GPU, three adapters co-resident
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --max-cpu-loras 16 \
  --lora-modules \
      support-general=./adapters/support-general \
      support-billing=./adapters/support-billing \
      support-escalation=./adapters/support-escalation \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port 8000
```

Flags that must be defended in `report.md`:

- `--max-lora-rank 64` — must be ≥ the largest adapter rank; if any adapter is `r=128`, bump this, which costs a one-time VRAM allocation.
- `--max-loras 4` — number of adapters the GPU can hold *active in the current step*. Low values force LRU eviction and increase tail latency when the traffic mix touches more adapters than the limit allows.
- `--max-cpu-loras 16` — bound on how many adapters stay hot in pinned CPU memory for fast re-activation. Lower = less RAM, more disk re-reads on eviction storms.

Request format (OpenAI-compatible):

```bash
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "support-billing",
  "prompt": "Customer writes: my last invoice shows ...",
  "max_tokens": 128,
  "temperature": 0
}'
```

The `"model"` field selects the adapter; omitting it routes to the base model (which is served alongside the adapters for free).

### 2.4 Variant B — Merged serve

Produce three merged checkpoints offline:

```python
# merge_all.py — one lossless merge per adapter
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import pathlib, torch

BASE = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTERS = ["support-general", "support-billing", "support-escalation"]
OUT = pathlib.Path("outputs/merged")
OUT.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(BASE)
for name in ADAPTERS:
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, f"./adapters/{name}")
    merged = model.merge_and_unload()           # fuses A·B into base weights, returns plain model
    out = OUT / name
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"merged {name} -> {out}")
```

Serve each as its own vLLM process (three ports, or three replicas behind a reverse proxy):

```bash
# serve_merged.sh — one process per tenant; run on separate GPUs or time-multiplex
for port in 8001 8002 8003; do
  case $port in
    8001) m=./outputs/merged/support-general ;;
    8002) m=./outputs/merged/support-billing ;;
    8003) m=./outputs/merged/support-escalation ;;
  esac
  python -m vllm.entrypoints.openai.api_server \
    --model "$m" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --port "$port" &
done
wait
```

For the comparison the merged variant is sized at **one shared GPU time-sliced across tenants** (apples to the Multi-LoRA single-GPU setup) and, separately, at **three dedicated GPUs** (what you would actually ship if every tenant demands peak throughput).

### 2.5 Load-test harness

The harness must drive both variants with the same traffic mix. Use `oha` (or `k6` or `vegeta`) to generate a Poisson-arrival request stream from a fixed prompt pool.

```python
# gen_traffic.py — writes requests.jsonl with a given per-adapter mix
import json, pathlib, random, argparse
random.seed(42)
ap = argparse.ArgumentParser()
ap.add_argument("--mix", nargs="+", default=["support-general:0.6",
                                              "support-billing:0.3",
                                              "support-escalation:0.1"])
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--prompt_file", default="./data/support_prompts.txt")
args = ap.parse_args()
prompts = [p for p in pathlib.Path(args.prompt_file).read_text().splitlines() if p.strip()]
mix = [(k, float(v)) for k, v in (m.split(":") for m in args.mix)]
total = sum(w for _, w in mix); mix = [(k, w/total) for k, w in mix]
out = []
for _ in range(args.n):
    r = random.random(); acc = 0.0
    for k, w in mix:
        acc += w
        if r <= acc:
            out.append({"model": k, "prompt": random.choice(prompts),
                        "max_tokens": random.randint(64, 256), "temperature": 0})
            break
pathlib.Path("requests.jsonl").write_text("\n".join(json.dumps(x) for x in out) + "\n")
```

Drive the Multi-LoRA server at `http://localhost:8000/v1/completions` and the merged tenants at a thin router that dispatches `"model": "support-general"` → `:8001`, etc. A 20-line FastAPI router is enough; `oha` gets a single URL and the router fans out.

Metrics to collect per variant:

| Metric | Why it matters |
|---|---|
| Throughput (req/s, tok/s) | Aggregate capacity on fixed hardware |
| TTFT p50 / p95 | User-visible first-token latency; sensitive to LoRA cold-swap events on Multi-LoRA |
| ITL p50 / p95 | Per-token latency; measures the Punica/Triton LoRA overhead |
| End-to-end p95/p99 | SLO verdict |
| Per-adapter latency | Isolates adapter-specific pathology (e.g. one adapter blows up decode) |
| Steady-state VRAM | Decides whether a bigger base (70B) still fits with N adapters |
| GPU utilization | Whether the Multi-LoRA server is bandwidth-bound or compute-bound |

Run a 60-second warmup, then 10-minute steady state with target QPS held constant. Repeat at three QPS levels per variant (half-capacity, near-saturation, past-saturation) to trace the latency-throughput curve.

### 2.6 Worked summary table (illustrative; numbers depend on hardware and traffic mix)

Reference numbers from running the pipeline on 1× A100-80GB, Llama-3.1-8B-Instruct base, 70/20/10 adapter mix, mean prompt 450 tokens, mean generation 128 tokens. Treat digits as illustrative; the examinable claim is the *shape* of the trade-off.

| Setup | Base VRAM | LoRA VRAM | Peak req/s | TTFT p95 | ITL p95 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Multi-LoRA (1 GPU, 3 adapters)       | 17 GB | 0.6 GB total | 62  | 240 ms | 18 ms | Shared base; 3-5% ITL overhead vs merged on same base |
| Merged × 3 tenants, 1 GPU, time-sliced | 17 GB × queued | 0 | 24 | 980 ms | 16 ms | Context-switching between merged models thrashes KV cache; dominated by reload |
| Merged × 3 tenants, 3 GPUs           | 17 GB each | 0 | 180 | 140 ms | 16 ms | Each tenant owns a GPU; no sharing cost; 3× hardware cost |
| Merged × 3 tenants, 3 GPUs, equal weight rebalanced by traffic | — | 0 | 180 (but idle on 2 GPUs 80% of the day) | 140 ms | 16 ms | The real bill is the idle share |

Patterns to read:

- **Multi-LoRA pays a small per-token overhead for a *large* reduction in base-weight VRAM.** 1 GPU handles all three tenants; the only question is whether 3-5% ITL is inside the SLO.
- **Merged on dedicated hardware wins on throughput and worst-case TTFT, but at 3× GPU cost.** This is the right answer only when at least one adapter demands peak capacity.
- **Merged on shared hardware is almost always wrong.** vLLM is not designed to hot-swap base-model checkpoints; reloading a 16 GB model between requests is seconds, not milliseconds.
- **The decision inverts at the long tail.** With 20+ low-QPS adapters, Multi-LoRA is the only option that is affordable; the merged-per-tenant bill goes linear with tenant count.

### 2.7 Directory layout (as delivered)

```
outputs/exercise-02/
├── env.txt
├── adapters/
│   ├── support-general/
│   ├── support-billing/
│   └── support-escalation/
├── outputs/merged/
│   ├── support-general/
│   ├── support-billing/
│   └── support-escalation/
├── serve_multi.sh
├── serve_merged.sh
├── merge_all.py
├── gen_traffic.py
├── router.py            # FastAPI, 20 lines, forwards to the right merged port
├── loadtest_multi.sh    # oha invocation
├── loadtest_merged.sh
├── results/
│   ├── multi/   (metrics.json, latency.histogram, nvidia-smi.csv)
│   ├── merged_shared/
│   └── merged_dedicated/
└── report.md
```

### 2.8 `report.md` skeleton

```markdown
# Multi-LoRA vs. Merged — report

## Setup
- Base: meta-llama/Llama-3.1-8B-Instruct @ bf16
- Adapters: 3; LoRA ranks 16/32/64; all target the same modules
- Hardware: 1× A100-80GB (Multi-LoRA, Merged-shared) and 3× A100-80GB (Merged-dedicated)
- Traffic: 70% support-general, 20% support-billing, 10% support-escalation
- Load generator: oha, Poisson arrivals, 10-min steady-state at 3 QPS targets
- vLLM: 0.6.3; --enable-lora, --max-loras 4, --max-lora-rank 64, --max-cpu-loras 16

## Results
(see table in §2.6; latency histograms in results/*.hist)

## Decision
For this adapter count, traffic mix, and SLO (TTFT p95 < 500 ms, ITL p95 < 25 ms):
**Multi-LoRA.**
- Fits 3 tenants on 1 GPU; VRAM headroom for 4-5 more adapters at the same base.
- TTFT and ITL inside SLO at the measured peak QPS.
- 3-5% ITL overhead vs merged is acceptable for the 2/3 VRAM reduction.

Switch to Merged-dedicated when one of:
- An adapter hits sustained > 50 req/s alone (Multi-LoRA saturates below the
  per-adapter capacity of a dedicated server).
- Rank climbs past the `max-lora-rank` ceiling; raising it reserves VRAM for
  everyone.
- Two adapters need independent eviction policies (regulated vs non-regulated).

Avoid Merged-shared. The hot-swap cost is unrecoverable on a single GPU.

## Risks / follow-ups
1. Measure the LoRA swap cost explicitly: synthetic burst into 10 adapters, watch
   the `--max-loras` eviction counter and TTFT tail.
2. Validate that Multi-LoRA respects isolation claims when one tenant is regulated
   (e.g. health-care adapter) — same physical process, same KV cache arena.
3. Re-run with 4-bit base weights (link to exercise 01); AWQ + Multi-LoRA shrinks
   base VRAM by ~3× and raises the adapter budget past 20.
```

## 3. Validation steps

1. **All adapters target the same base.** Every `adapter_config.json` has the same `base_model_name_or_path` and the same `target_modules` list. Any mismatch is rejected at vLLM startup; fix before continuing.
2. **`--max-lora-rank` ≥ max adapter rank.** Grep each `adapter_config.json` for `r:`; the serve flag must cover the largest.
3. **One shared base; N adapter loads.** `nvidia-smi` on the Multi-LoRA server shows a single base-weight allocation and small per-adapter allocations proportional to rank; no duplicate base memory.
4. **The Merged-dedicated setup is three *isolated* processes.** `ps -o pid,pcpu,rss,command` lists three distinct vLLM workers with separate GPU bindings; no shared Python process.
5. **Traffic mix is enforced, not nominal.** `gen_traffic.py` output is counted; the observed per-adapter request share matches the declared mix within ±2%.
6. **TTFT and ITL are reported as distinct percentiles.** A combined "p95 latency" is not enough — both numbers are required because they diagnose different bottlenecks.
7. **At least one past-saturation QPS point is measured.** The curve must show where latency knee-points, not just a mid-range measurement.
8. **Per-adapter latency is broken out.** If `support-escalation` has higher ITL under Multi-LoRA, the report names why (larger rank? colder adapter?).
9. **Cost model is present.** `report.md` includes a $-per-request estimate (GPU $/hr × 1/req-per-s) for both variants at steady state; the decision references cost explicitly, not just latency.
10. **Merge is lossless.** A sanity check runs the merged model and the (base+adapter) stack through 50 identical prompts; token-by-token output matches up to sampling-temperature=0 determinism.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Correct deployment setup** | Adapters do not co-load; or merged tenants not actually merged (still carry PEFT wrapper) | Both variants deploy but flags default and max-lora-rank is below a used adapter's rank | `--enable-lora`, correct `--max-lora-rank`, `--max-loras`, `--max-cpu-loras`; `merge_and_unload()` sanity-checked; separate ports for merged tenants |
| **Traffic generator and SLO definition** | Random requests with no declared mix or SLO | Mix declared but not enforced; one SLO without TTFT/ITL split | Declared+observed per-adapter mix; SLO expressed as TTFT and ITL percentiles plus throughput target |
| **Latency + throughput comparison** | Single aggregate number; no percentiles | Percentiles reported but no past-saturation point | TTFT p50/p95/p99, ITL p50/p95/p99, throughput curve across 3 QPS levels per variant |
| **Resource + cost accounting** | VRAM and GPU-hr not reported | VRAM reported but $-per-request missing | Steady-state VRAM per variant + GPU-hr-per-request + a 1–year cost delta at the declared traffic volume |
| **Decision + inversion cases** | Picks one variant with no caveats | Picks a variant and lists one caveat | Picks a variant, defends it on cost+latency+VRAM, names the exact traffic shift or SLO change that flips the decision |

**Pass threshold:** 7/10, with no dimension at 0 and the latency-comparison dimension at 2.

## 5. Common mistakes

- **Serving adapters with mismatched base_model_name_or_path.** vLLM refuses to co-load them; the fix is upstream in the training config, not in the serving flags.
- **Setting `--max-lora-rank` below the highest adapter rank** and only finding out at request time with a cryptic rank-mismatch error. Validate the config before starting the server.
- **Comparing Multi-LoRA on 1 GPU to Merged on 3 GPUs and declaring Multi-LoRA slower.** Equalize hardware, or compare cost-per-request instead of raw throughput.
- **Not measuring past-saturation.** Multi-LoRA and merged have different knees; reporting only mid-range QPS hides the SLO breakpoint.
- **Treating `merge_and_unload()` as lossy.** It is an algebraic fuse; the difference vs the unmerged graph is numerically negligible at bf16. Document the sanity-check, do not rely on folklore.
- **Running the Multi-LoRA benchmark without pinning `--max-loras`.** Default values vary across vLLM versions; the knob is central to the exercise and must be set explicitly and recorded.
- **Ignoring the `max_loras_per_step` constraint.** vLLM batches requests; if 12 requests in flight touch 8 adapters, the batch is split. Record which batches are LoRA-split in the server logs and explain any TTFT tail that correlates with the split.
- **Reporting only aggregate latency.** The headline risk of Multi-LoRA is *per-adapter tail* latency on cold adapters; aggregate numbers mask it.
- **Merging and then trying to A/B test.** A merged tenant is a snapshot; A/B rollback requires redeploying a different merged checkpoint. If product wants fast A/B, pin on Multi-LoRA for the A/B phase and merge for GA.
- **Underestimating KV-cache contention under Multi-LoRA.** All adapters share the arena; a long-context request on one tenant shrinks capacity for all. If one tenant's context is dramatically larger than the others, consider moving it to Merged-dedicated.
- **Not pricing out the alternative.** Picking Multi-LoRA because "it's cheaper" without a $-per-request number is not a defense. Price the GPU-hr at the lab's actual cloud rate and multiply by observed throughput.

## 6. References

- vLLM documentation — *Multi-LoRA* (serving flags `--enable-lora`, `--max-loras`, `--max-lora-rank`, `--max-cpu-loras`, `--lora-modules`). <https://docs.vllm.ai/en/latest/models/lora.html>
- Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen — *LoRA: Low-Rank Adaptation of Large Language Models*. <https://arxiv.org/abs/2106.09685>
- Sheng, Cao, Li, Hooper, et al. — *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*. <https://arxiv.org/abs/2311.03285>
- Hugging Face PEFT documentation — `PeftModel.merge_and_unload()`. <https://huggingface.co/docs/peft/>
- vLLM — *PagedAttention* architecture, KV-cache arena semantics. <https://docs.vllm.ai/en/latest/design/arch_overview.html>
- `oha` load tester. <https://github.com/hatoo/oha>
- NIST AI Risk Management Framework — *Govern 1.1* (explicit roles for serving decisions) and *Manage 2.1* (performance monitoring in deployment). <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — *ML05 Model Theft* and *ML06 AI Supply Chain Vulnerabilities* apply to tenant isolation on Multi-LoRA and the pinning of adapter artifacts. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — technique catalog; multi-tenant isolation gaps and adapter-level data leakage map to the *Exfiltration via ML Inference API* techniques and should be included in the risk section of `report.md`. <https://atlas.mitre.org/>
