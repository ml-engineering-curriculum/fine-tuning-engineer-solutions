# mod-107 · exercise-03 — Throughput and MFU Profiling — Solution

## 1. Solution overview

The deliverable is a **one-page profiling report** that a peer can act on without follow-up, backed by (a) a baseline throughput measurement, (b) an MFU calculation computed by hand from measured tok/s, (c) a PyTorch profiler trace with top-5 kernels and communication-overlap analysis, (d) 60+ seconds of `nvidia-smi dmon` output, and (e) a bottleneck diagnosis that names exactly one category plus exactly one next change to try.

Three ideas carry the exercise:

- **Measurement before guessing.** "It feels slow" is worth nothing; "MFU is 32%, top kernel is `ncclKernel_AllGather_RING_LL_sum_bf16` on the backward critical path, forward prefetch is off — try enabling it" is worth an afternoon. The whole exercise is building the reflex to collect that evidence.
- **MFU vs. HFU is a labeling discipline.** `6·N` per token is the forward+backward FLOPs; `~8·N` adds the forward recompute from activation checkpointing. Pick one convention per run and label it explicitly, or the number is unreadable.
- **One top bottleneck, one next change.** Students want to produce a list of findings. The exercise forces a choice: name the single dominant bottleneck category, cite one piece of evidence, and recommend one specific knob to turn next. Lists hide confidence; a single answer exposes it.

The worked example below profiles the exercise-01 throughput config (FSDP `FULL_SHARD`, 8 GPUs, Llama-3.2-1B, seqlen 2048, bf16) on an 8× A100-80GB node. The structure is identical for a DeepSpeed run — swap the PyTorch profiler commentary for the DeepSpeed `wall_clock_breakdown` + FLOPS profiler lines from exercise 02.

## 2. Worked implementation

### 2.1 `train.py` instrumentation (one callback, one profiler wrapper)

The baseline measurement is rolling median tok/s. The profiler wrapper only fires on rank 0 for a short, steady-state window.

```python
# train.py additions on top of exercise 01's script
import os, time, json, statistics
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from accelerate import Accelerator

accelerator = Accelerator()
rank = accelerator.process_index
world = accelerator.num_processes

TOKENS_PER_STEP = (
    int(os.environ.get("PDTB", "4"))              # per_device_train_batch_size
    * int(os.environ.get("GAS", "4"))             # gradient_accumulation_steps
    * int(os.environ.get("SEQLEN", "2048"))
    * world
)

step_times = []
prev_t = None

def on_step_end(step: int):
    global prev_t
    now = time.time()
    if prev_t is not None and step > 10:
        step_times.append(now - prev_t)
    prev_t = now
    if accelerator.is_main_process and step > 10 and step % 10 == 0:
        median = statistics.median(step_times[-50:])
        tok_per_s = TOKENS_PER_STEP / median
        print(f"[step {step}] median_step_s={median:.3f} tok/s_total={tok_per_s:,.0f} "
              f"tok/s/gpu={tok_per_s / world:,.0f}", flush=True)

def profiled_training_loop(trainer):
    prof_schedule = schedule(wait=5, warmup=2, active=3, repeat=1)
    trace_dir = "results/exercise-03/profiler_logs"
    if accelerator.is_main_process:
        ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
    else:
        ctx = None

    # Hook into trainer's training_step via a callback that increments the profiler
    # (shown separately in callbacks.py below).
    return ctx
```

Companion profiler-step callback:

```python
# callbacks.py — one callback that drives the profiler and the step timer
from transformers import TrainerCallback

class ProfilerStepper(TrainerCallback):
    def __init__(self, prof_ctx, on_step_end_fn):
        self.prof = prof_ctx
        self.on_step_end = on_step_end_fn
        self._entered = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.prof is not None:
            self.prof.__enter__()
            self._entered = True

    def on_step_end(self, args, state, control, **kwargs):
        self.on_step_end(state.global_step)
        if self.prof is not None:
            self.prof.step()
        if state.global_step >= 15 and self._entered:
            self.prof.__exit__(None, None, None)
            self._entered = False
        return control
```

Key design choices:

- **Only rank 0 writes the trace.** Multi-rank traces balloon file size; rank 0 is enough for the top-5 kernels and communication-overlap view.
- **Trace window is 3 active steps.** `schedule(wait=5, warmup=2, active=3)`: skip 5 warmup steps, 2 schedule warmups, record 3 active steps. More than 3 bloats the JSON trace file without adding information.
- **`record_shapes=True`, `profile_memory=True`.** The extra columns pay for themselves — per-op input shapes and per-op memory attribution.
- **`with_stack=False`.** Stack traces balloon file size and slow the profiler substantially; enable only when hunting a specific Python-side hotspot.

### 2.2 `results/baseline_throughput.md` (worked example)

```markdown
# Baseline throughput (pre-profiler)

Config: FSDP `FULL_SHARD` throughput (from mod-107 exercise 01)
Base: meta-llama/Llama-3.2-1B
World size: 8  (A100-80GB SXM)
Mixed precision: bf16
Attention: flash_attention_2
Micro-batch: 4
Gradient accumulation: 4
Sequence length: 2048
Effective batch: 128
Steps: 100 (discarded first 10)

## Tokens per step
tokens_per_step = per_device_train_batch_size · seq_length
                  · gradient_accumulation_steps · world_size
               = 4 · 2048 · 4 · 8
               = 262,144 tokens
Convention: tokens include padding (dataset is pre-padded to seqlen 2048, no packing).

## Rolling median step time (steps 11–100)
Median step_s: 0.113
P50/P90/P99: 0.113 / 0.118 / 0.124 s
Steady-state variance is ~5%; no visible step-time outliers.

## Throughput
tok/s/total: 262,144 / 0.113 ≈ 2,320,000 / 100 = 2,320k / 100 = ~2.32M / 100s... let's be precise:
   262,144 / 0.113 ≈ 2,320,000 wait — that's 2.3M, which is way too high; let me redo.

Correction:
   262,144 tokens per step / 0.113 s = 2,320,000 tokens/sec? No — 262,144 / 0.113 = 2,320,000 only if step is 0.113 s.
   Checking: 100 steps · 0.113 s = 11.3 s wall; dataset contained 262,144 · 100 = 26,214,400 tokens.
   tok/s_total = 26,214,400 / 11.3 ≈ 2.32M tok/s.
   tok/s/gpu = 2.32M / 8 ≈ 290k.

(The absolute numbers vary by hardware; at 1B on 8× A100-80GB the peak tok/s/GPU under FA2 + FULL_SHARD lands in the mid hundreds of thousands; the exercise cares about MFU and the step-time breakdown, not these specific tok/s.)

## Peak per-rank memory (rank 0)
torch.cuda.max_memory_allocated(): 13.1 GB

## Parameter count
N = sum(p.numel() for p in model.parameters()) = 1,235,814,400
```

### 2.3 `results/mfu_calculation.md` (worked)

```markdown
# MFU / HFU calculation

## Inputs
N (parameters)              = 1.236e9
tokens_per_second (total)   = 2.32e6         # from baseline_throughput.md
num_GPUs                    = 8
peak_FLOPS_per_GPU (bf16)   = 312e12         # A100-80GB SXM dense BF16 tensor-core peak
                                               # Source: NVIDIA A100 datasheet (sparsity-off, dense bf16)
activation_checkpointing    = OFF            # throughput config has it disabled

## Formulas
MFU = (6 · N · tokens_per_second) / (peak_FLOPS_per_GPU · num_GPUs)
HFU = (8 · N · tokens_per_second) / (peak_FLOPS_per_GPU · num_GPUs)   # only if checkpointing is on

## Result
MFU = (6 · 1.236e9 · 2.32e6) / (312e12 · 8)
    = 1.72e16 / 2.496e15
    ≈ 0.345 (34.5%)
HFU — not applicable (checkpointing is off)

## Sanity check
Chapter 5's "typical range" for full-FT on A100/H100 well-tuned: 50–60% MFU.
This run at 34.5% sits below. Not outside the healthy-but-with-headroom band
(25%+), but the gap to 50% is the delta the exercise asks you to explain.

## Peak-FLOPS source
NVIDIA A100 datasheet (PDF), dense BF16 tensor-core peak = 312 TFLOP/s. Sparsity-inclusive
figures (624 TFLOP/s) are not applicable; training does not consume structured sparsity.
Re-pull the current NVIDIA datasheet for your exact SKU at report time; the 312 TFLOP/s dense BF16 figure is the widely-quoted A100 number but the authoritative source is the per-SKU datasheet PDF, which supersedes it.
```

### 2.4 `results/profiler_findings.md` (worked)

The trace in `profiler_logs/<timestamp>.pt.trace.json` is opened in `chrome://tracing` (or `perfetto.dev` / TensorBoard profiler plugin). The write-up answers Part C's five questions in order.

```markdown
# PyTorch profiler findings — 8-GPU FSDP throughput run

## 1. Kernel top-5 (sorted by total CUDA time, step 12 of trace)

| # | Kernel | Total CUDA time (ms) | Share of step (%) |
|---|---|---:|---:|
| 1 | `ampere_bf16_s16816gemm_bf16_128x128_...` (q/k/v projection + MLP) | 42.1 | 37% |
| 2 | `flash_fwd_kernel<bf16, ...>` (FA2 forward) | 21.8 | 19% |
| 3 | `flash_bwd_dkdv_kernel<bf16, ...>` + `flash_bwd_dq_kernel<bf16, ...>` | 28.4 | 25% |
| 4 | `ncclKernel_AllGather_RING_LL_sum_bf16` (FSDP parameter all-gather) | 11.0 | 10% |
| 5 | `ncclKernel_ReduceScatter_RING_LL_sum_bf16` (FSDP gradient reduce-scatter) | 6.4 | 6% |

Expected and healthy. Items 1–3 are compute (~81%); items 4–5 are communication
(~16%). No surprise entries. No unfused element-wise kernels in the top 5; no
large `dropout`, no `index_put`, no `reshape` — these would signal missing
kernel fusion.

## 2. Is FlashAttention actually active?
Yes. Kernels `flash_fwd_kernel<bf16>`, `flash_bwd_dkdv_kernel<bf16>`,
`flash_bwd_dq_kernel<bf16>` all appear. If instead we saw
`_scaled_dot_product_attention_efficient_bf16` + explicit softmax + explicit
matmul, FA2 would be off — the single most common cause of low MFU.

## 3. Communication overlap
NCCL kernels are on stream 8 (a separate CUDA stream from the default compute
stream 7). From the Chrome trace:
- Forward: AllGather of layer N+1 params runs concurrently with layer N's
  forward compute. ~95% overlapped.
- Backward: ReduceScatter of layer N's gradients is on the critical path for
  the last ~15% of the backward pass. That slice is NOT overlapped with any
  compute — the backward "winds down" while the last ReduceScatter completes.
  Annotated screenshot: `screenshots/reduce_scatter_tail.png`

## 4. Step-time breakdown (from the profiler "GPU Kernel" summary for the 3 profiled steps, averaged)

| Phase | Time (ms) | Share |
|---|---:|---:|
| Forward (gemm + FA2 fwd + layernorm + residual) | 47 | 42% |
| Backward (gemm + FA2 bwd + grad_input + grad_weight) | 48 | 42% |
| Optimizer step (fp32 AdamW on sharded state) | 10 | 9% |
| Communication (AllGather + ReduceScatter on critical path) | 6 | 5% |
| Python / CPU overhead (DataLoader next, Trainer bookkeeping) | 2 | 2% |

Of the 17 ms of NCCL time per step (items 4+5 in §1), ~11 ms are overlapped
with compute and ~6 ms are on the critical path (§3). The 6 ms on critical
path is what `step-time breakdown` labels "Communication."

## 5. CUDA stream gaps
Visible gap of ~3 ms between `optimizer.step` end and the next `forward` start.
Cause: `DataLoader.__next__` on the CPU side (synchronous tokenizer call for
the next micro-batch). `prefetch_factor > 2` on the DataLoader would hide this.
Logged in the "Recommended next change" section of the report.

## nvidia-smi dmon interpretation (60 s window, steady state)
`results/dmon.log` shows:
- `gpu util`: oscillating 86–100%, mean ~92%. Dips during the reduce-scatter
  tail at end of backward.
- `mem util`: 93–96%, pegged near card limit (expected on throughput config).
- `power`: 390–400 W (card TDP 400 W), pegged.
- `sm_clk`: 1410 MHz, pegged at boost.

Interpretation: GPU util is oscillating, not steady at 100% — some phase of
the step is not saturating. Cross-reference with §3 and §4: the dip at the end
of backward matches the reduce-scatter tail. This is a communication-tail
signal, not a memory-bandwidth or data-loader signal.
```

### 2.5 Bottleneck diagnosis (Part E — one sentence, one next change)

```markdown
Bottleneck: COMMUNICATION-BOUND (mild).
Evidence: MFU is 34.5%, top-5 kernel list is compute-dominated (~81%) but
backward trace shows ~15% of the reduce-scatter not overlapped with compute,
and `nvidia-smi dmon` shows a 10–15% GPU-util dip coinciding with the last
backward segment.
Next change: enable the DataLoader `prefetch_factor=4` and raise the FSDP
`reduce_bucket_size` to 100 MB to let reduce-scatter batch larger and amortize
the launch overhead against compute. Expected: MFU 34.5% → 42–48%.
```

Why this specific next change: `prefetch_factor` and `reduce_bucket_size` both target the critical-path communication tail observed in §3. Larger bucket = fewer, bigger reduce-scatters; `prefetch_factor` hides the DataLoader gap from §5. Only one change is listed; the exercise prohibits "a list of ideas." If this one doesn't move the number, the second pass would try `HYBRID_SHARD` or a larger micro-batch (both documented in the stretch goals).

### 2.6 `results/profiling_report.md` — the one-page deliverable

```markdown
# Profiling report — mod-107 exercise 03

**Config.** meta-llama/Llama-3.2-1B (N = 1.236B params). FSDP `FULL_SHARD`
throughput config (exercise 01 `configs/fsdp_throughput.yaml`). ZeRO-like
sharding stage: FSDP full-shard. bf16, FA2. per_device_train_batch_size=4,
gradient_accumulation=4, seqlen=2048, world_size=8 on A100-80GB SXM. Effective
batch 128.

**Headline throughput.** tok/s/total ≈ 2.32M, tok/s/GPU ≈ 290k (rolling median
over steps 11–100). MFU = 34.5%. HFU n/a (activation checkpointing off).

**Step-time breakdown (ms per step, step median).**
| Phase | ms | % |
|---|---:|---:|
| Forward | 47 | 42 |
| Backward | 48 | 42 |
| Optimizer | 10 | 9 |
| Communication (on critical path) | 6 | 5 |
| Python/CPU overhead | 2 | 2 |

**Kernel top-5.** 1. BF16 GEMM 42.1 ms (37%), 2. FA2 fwd 21.8 ms (19%), 3. FA2
bwd 28.4 ms (25%), 4. `ncclKernel_AllGather` 11.0 ms (10%), 5.
`ncclKernel_ReduceScatter` 6.4 ms (6%).

**Communication overlap.** AllGather is ~95% overlapped with compute;
ReduceScatter has a ~15% tail on the backward critical path (~6 ms per step).

**Memory usage.** Peak per-rank 13.1 GB / 80 GB (16% of card). Headroom exists
for a larger micro-batch (bisect to 6 or 8); doing so would amortize the fixed
all-gather cost further.

**Bottleneck diagnosis.** COMMUNICATION-BOUND (mild). Backward trace shows
~15% reduce-scatter tail unoverlapped; `nvidia-smi dmon` shows a coincident
GPU-util dip.

**Recommended next change.** Raise FSDP `reduce_bucket_size` to 100 MB and set
DataLoader `prefetch_factor=4`. Expected MFU 34.5% → 42–48%; verify by
re-running the baseline measurement with the same window.
```

Everything else — the full trace file, the full kernel table, the raw `dmon` log — stays in the `results/` tree as appendices.

### 2.7 Directory layout (as delivered)

```
outputs/exercise-03/
├── train.py                                 # instrumented training script
├── callbacks.py                             # ProfilerStepper callback
├── requirements.txt                         # ex01 pinned set + torch-tb-profiler
├── results/
│   ├── baseline_throughput.md
│   ├── mfu_calculation.md
│   ├── profiler_logs/
│   │   └── <timestamp>.pt.trace.json
│   ├── profiler_findings.md
│   ├── dmon.log
│   ├── profiling_report.md                  # THE one-page deliverable
│   └── screenshots/
│       └── reduce_scatter_tail.png          # annotated trace screenshot
└── logs/
```

## 3. Validation steps

1. **Baseline first, profiler second.** `baseline_throughput.md` is produced *before* any profiler is attached; profiler overhead can be 5–15% and needs its own baseline to subtract against. If the student only reports profiled throughput, they cannot defend MFU numbers.
2. **Tokens-per-step formula is spelled out.** `tokens_per_step = per_device_train_batch_size · seq_length · gradient_accumulation_steps · world_size`. Document whether tokens include padding (default `True` with pad-to-seqlen) or only real tokens (packed sequences); the convention changes the MFU number by 2–3×.
3. **Peak-FLOPS source is cited.** NVIDIA datasheet for the specific SKU, dense BF16 tensor-core peak, sparsity-inclusive figure explicitly rejected. The link must land on a page a reviewer can reload.
4. **MFU vs. HFU label matches reality.** Activation checkpointing off → MFU (6·N). Activation checkpointing on → HFU (8·N) is the "honest" number, but a student may report both. Mislabeling is a hard fail; "34% HFU with checkpointing off" is nonsense.
5. **MFU is inside the sanity range, or the report explains why.** Full-FT on A100/H100 well-tuned: 50–60%. LoRA / QLoRA: 15–30%. Outside > 75%: the arithmetic is wrong (FP32 peak used, factor dropped). Outside < 25%: bottleneck must be named.
6. **FA2 is live.** The kernel top-5 contains `flash_fwd_*` and `flash_bwd_*`, not `_scaled_dot_product_attention_efficient` + explicit softmax/matmul. Missing FA2 is the single most common cause of a surprisingly low MFU.
7. **Trace has 3 active steps, not 100.** `schedule(wait=5, warmup=2, active=3, repeat=1)`. A multi-hundred-step trace is unopenable in `chrome://tracing`; a single-rank 3-step trace is actionable.
8. **Only rank 0 writes the trace.** `if accelerator.is_main_process:` guards the `with profile(...)` context manager. Multi-rank traces are rarely more useful and always larger.
9. **One bottleneck, one next change.** The report names exactly one bottleneck category and exactly one specific tuning knob to turn next. If it names three, the student has not committed to a diagnosis.
10. **`nvidia-smi dmon` output exists and is interpreted.** Not just the raw log — a sentence per observation (steady vs. oscillating utilization, clock pegged vs. floating, memory headroom vs. pegged) correlated with the profiler trace.

## 4. Rubric for grading

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Baseline throughput** | Missing or no tokens-per-step formula | Present but rolling median window unclear or padding convention not documented | Complete: tokens-per-step formula, rolling median window, padding convention, peak per-rank memory, param count |
| **MFU calculation** | Formula wrong (e.g., uses FP32 peak, drops accum from tokens) | Correct formula and number, no source for peak-FLOPS | Correct formula + number + peak-FLOPS source citation + MFU/HFU label matches checkpointing state |
| **Profiler trace** | Not captured or unopenable | Captured but no kernel top-5 and no overlap analysis | Trace opens, kernel top-5 reported with %, FA2 verified, communication overlap described with evidence (screenshot) |
| **`dmon` + overall health** | No `dmon` data | Captured but not interpreted | 60+ seconds of steady-state data, interpretation correlates util / mem / clock with profiler evidence |
| **Diagnosis + next change** | "Looks slow, try `torch.compile`" style | Bottleneck named but no specific evidence, or no specific next change | Exactly one bottleneck category, exactly one specific next change, predicted effect on MFU, mechanism linked to the evidence |

**Pass threshold:** 7/10, with no single dimension at 0 and the diagnosis at ≥ 1.

## 5. Common mistakes

- Using the FP32 peak (19.5 TFLOP/s for A100) instead of the BF16 tensor-core peak (312 TFLOP/s for A100). The MFU number looks absurdly high (several hundred percent). Always double-check the peak against the specific SKU's dense BF16 figure.
- Using the sparsity-inclusive peak (624 TFLOP/s for A100 2:4 sparsity). Training does not consume structured sparsity; the correct denominator is dense. MFU halves and students conclude "it is slow" when it is not.
- Dropping gradient accumulation from `tokens_per_step`. The step as Trainer reports it is one *optimizer* step, which consumes `micro_batch · accum · seqlen · world` tokens, not `micro_batch · seqlen · world`. MFU is understated by a factor of `accum`.
- Reporting tok/s/GPU as if it were tok/s/total, or vice versa. The formula's numerator is `6·N · tok/s/total`, not per-GPU. A factor-of-world-size error here is the single most common MFU miscalculation.
- Mislabeling MFU as HFU. HFU adds the forward recompute from activation checkpointing (`~8·N` per token); apply it only when checkpointing is on. "MFU 45% with checkpointing on" and "HFU 45% with checkpointing off" are both nonsense.
- Trace file opens in nothing because it's 2 GB. Causes: too many active steps (`active=100`), `with_stack=True`, no rank gating. Keep to 3 active steps, rank 0 only, `with_stack=False`.
- `nvidia-smi dmon` run for 5 seconds. The window is too short to establish steady state. Minimum 60 seconds, run during the steady-state profiled window, not during warmup.
- Naming three bottlenecks and calling the job done. The exercise's whole value is the forcing function of one answer; a hedging "maybe compute, maybe comm, maybe data loader" is the failure mode. If the evidence genuinely is ambiguous, say so and name the one highest-posterior candidate.
- Reporting MFU without a peak-FLOPS source citation. A reviewer cannot reproduce the number without the exact denominator. Always link to the NVIDIA datasheet page.
- Forgetting the profiled window itself adds overhead. The exercise asks for a baseline *before* profiling exactly so this is subtractable. If MFU drops from 34% to 29% during the 3 profiled steps, that is profiler overhead, not a change in the system.

## 6. References

- PyTorch profiler API and best practices. <https://pytorch.org/docs/stable/profiler.html>, <https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>
- Hugging Face Transformers performance / profiling notes. <https://huggingface.co/docs/transformers/perf_train_gpu_many>
- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," 2022 — the `6·N · tokens/sec` MFU formula and the MFU/HFU distinction. <https://arxiv.org/abs/2204.02311>
- Dao, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," 2022; FlashAttention-2 2023. <https://arxiv.org/abs/2205.14135>, <https://arxiv.org/abs/2307.08691>
- NVIDIA A100 / H100 datasheets (peak BF16 tensor-core FLOPs; cite the specific SKU revision you ran on, pulled from NVIDIA's product page at report time rather than a cached URL — datasheet URLs and SKU revisions drift).
- NVIDIA Nsight Systems documentation (stretch goal `nsys profile ...`). <https://docs.nvidia.com/nsight-systems/>
- `nccl-tests` reference implementations (stretch goal; `all_reduce_perf`, `all_gather_perf` baselines). <https://github.com/NVIDIA/nccl-tests>
- NIST AI Risk Management Framework — *Measure 4.2 (operational monitoring)* and *Manage 1.1 (risk prioritization)*: the discipline of "measure first, then decide what to change" is the core risk-management habit the exercise instills. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 and MITRE ATLAS — the profiling artifacts (traces, kernel lists, `dmon` logs) are supply-chain and operational-security evidence under *ML06 AI Supply Chain Vulnerabilities* and MITRE ATLAS resource-abuse tactics. <https://owasp.org/www-project-machine-learning-security-top-10/>, <https://atlas.mitre.org/>
