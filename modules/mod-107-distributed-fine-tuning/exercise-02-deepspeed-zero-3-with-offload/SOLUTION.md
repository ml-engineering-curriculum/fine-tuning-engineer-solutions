# mod-107 · exercise-02 — DeepSpeed ZeRO-3 with CPU Offload — Solution

## 1. Solution overview

The deliverable is three DeepSpeed JSON configs, one sweep CSV, a cross-framework comparison table, a FLOPS-profiler + wall-clock-breakdown artifact, and a one-page recommendation that answers "which framework and offload level would you pick for scenario X?" Framework-vs-framework answers in the abstract are worthless; the exercise's whole point is the reasoning for a *specific* job.

Three ideas carry the exercise:

- **Offload is a feasibility lever, not a throughput lever.** The expected throughput cost curve is roughly `+0% → −10–30% → −2–5×` across *(no offload → optimizer offload → full offload)*. If the cost curve looks different, either the host is RAM-contended (watch `free -h` during the run) or something in the recipe moved between runs.
- **Framework parity is enforced by loss, not by vibes.** The same recipe (same base, tokenizer, data, effective batch, LR schedule, seqlen, epochs) must produce final loss within ~1% across FSDP `FULL_SHARD` and DeepSpeed ZeRO-3 no-offload. A larger gap means the recipe moved; reconcile that before claiming a throughput number.
- **`zero3_init_flag: true` and `"auto"` fields are non-optional, not stylistic.** Without `zero3_init_flag`, `from_pretrained` materializes the full model in host RAM on *every* rank before sharding — ~48 GB × world_size, enough to OOM a 256 GB host at 8 ranks with a 7B base. Without `"auto"` on the batch/clip fields, Trainer and DeepSpeed disagree on `train_batch_size = micro · accum · world` and the launcher errors.

The worked configs below target an 8× A100-80GB single node; 4-GPU numbers are included as the documented fallback. Host RAM assumption: ≥256 GB (verify with `free -h` before starting; for a 7B base, optimizer offload needs ~48 GB of FP32 AdamW state resident).

## 2. Worked implementation

### 2.1 `configs/ds_config_z3_noffload.json` — ZeRO-3, no offload (direct FSDP counterpart)

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

Commentary:

- `stage3_max_live_parameters` + `stage3_max_reuse_distance` are the two knobs that control how many parameter shards stay resident per step; defaults are tuned for 7B. Lower them if a tight memory budget forces it, higher them (up to `1e10`) if you have headroom and want fewer prefetch misses.
- `reduce_bucket_size: "auto"` lets DeepSpeed pick based on hidden size; this is the single highest-value "auto" field and the most common misstep when overridden manually.
- `wall_clock_breakdown: false` here keeps the no-offload sweep fast; flip to `true` only for the Part-C profiling pass.

### 2.2 `configs/ds_config_z3_optoffload.json` — ZeRO-3 + optimizer offload

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000,
    "stage3_gather_16bit_weights_on_model_save": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

`pin_memory: true` is non-optional for throughput — unpinned CPU → GPU transfers go through a bounce buffer and double the already-painful host-GPU round trip at the optimizer step.

### 2.3 `configs/ds_config_z3_full_offload.json` — ZeRO-3 + optimizer + parameter offload

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000,
    "stage3_gather_16bit_weights_on_model_save": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

Full offload is the "fit at all costs" config. Expect 2–5× step time vs. no-offload; frame it as feasibility, not throughput.

### 2.4 `configs/deepspeed_accelerate.yaml` (one file, templatized via env var)

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: bf16
num_processes: 8
use_cpu: false
machine_rank: 0
main_training_function: main
rdzv_backend: static
deepspeed_config:
  deepspeed_config_file: ${DS_CONFIG_FILE}   # set at launch time
  zero3_init_flag: true
  zero3_save_16bit_model: true
  deepspeed_multinode_launcher: standard
```

Launcher (switch `DS_CONFIG_FILE` per run):

```bash
DS_CONFIG_FILE=configs/ds_config_z3_noffload.json \
  accelerate launch \
    --config_file configs/deepspeed_accelerate.yaml \
    --num_processes 8 \
    train.py

DS_CONFIG_FILE=configs/ds_config_z3_optoffload.json \
  accelerate launch --config_file configs/deepspeed_accelerate.yaml --num_processes 8 train.py

DS_CONFIG_FILE=configs/ds_config_z3_full_offload.json \
  accelerate launch --config_file configs/deepspeed_accelerate.yaml --num_processes 8 train.py
```

The `train.py` from exercise 01 is unchanged; DeepSpeed takes over via the Accelerate config. Verify by watching the launcher log for `DeepSpeed info: engine_version=...` before step 0.

### 2.5 FLOPS profiler + wall-clock-breakdown overlay (no-offload run only)

Add these two blocks to `ds_config_z3_noffload.json` for the Part-C profiling pass only:

```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 10,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    "output_file": "results/exercise-02/flops_profile_noffload.txt"
  },
  "wall_clock_breakdown": true
}
```

`profile_step: 10` picks a steady-state step (post-warmup). `top_modules: 3` keeps the output readable; `detailed: true` adds per-module MACs and param counts that matter for the per-token FLOPs cross-check.

### 2.6 Expected sweep (worked numbers for an 8× A100-80GB node, Llama-3.2-1B, seqlen=2048, effective batch 128)

These are *illustrative* calibration points; the ratios and the shape of the cost curve are the examinable claim, not the absolute digits, which are hardware-specific and will shift on a different SKU.

| Run | Config | Peak GPU mem/rank (GB) | Peak host RAM (GB) | Tok/s/GPU | Tok/s total | Final loss | Status |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | FSDP `FULL_SHARD` throughput (from ex01) | ~13 | ~30 | ~1.80k | ~14.5k | 1.42 | clean |
| 2 | DeepSpeed ZeRO-3 no offload | ~12 | ~34 | ~1.70k | ~13.6k | 1.42 | clean |
| 3 | DeepSpeed ZeRO-3 + optimizer offload | ~7 | ~82 | ~1.35k | ~10.8k | 1.42 | clean |
| 4 | DeepSpeed ZeRO-3 + optimizer + param offload | ~4 | ~130 | ~0.45k | ~3.6k | 1.42 | clean |
| 5 | DeepSpeed ZeRO-2 no offload | ~22 | ~34 | ~1.85k | ~14.8k | 1.42 | clean at 1B; OOM expected at 7B |

**Reading the table**:

- Rows 1 and 2 agree on memory and throughput within ~5%. If a student's numbers diverge by >10%, the recipe moved (micro-batch? LR schedule? seqlen?) — reconcile before comparing.
- Row 3 vs. 2: optimizer offload costs ~20% throughput, saves ~40% GPU memory, adds ~48 GB host RAM for the FP32 AdamW state (12 bytes/param · 1.24B params ≈ 15 GB in this 1B example; multiply by the 7B factor for a real production run).
- Row 4 vs. 2: full offload costs ~4× throughput. The value proposition is "it fits," not "it's fast."
- Row 5: at 1B, ZeRO-2 is fine and slightly faster than ZeRO-3 (less all-gather traffic). At 7B on 8× 80GB it still fits; at 13B+ or on 40 GB cards it OOMs — document "OOM as expected" and move on.

### 2.7 `results/framework_comparison.md` (worked table)

```markdown
# Framework comparison (8× A100-80GB, Llama-3.2-1B base, effective batch 128)

| Row | Config | GPU mem/rank (GB) | Host RAM (GB) | Tok/s/GPU | Speedup vs 1-GPU baseline | Final loss | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | FSDP `FULL_SHARD` throughput (ex01) | 13 | 30 | 1.80k | 6.5× | 1.42 | Baseline for comparison |
| 2 | DeepSpeed ZeRO-3, no offload | 12 | 34 | 1.70k | 6.2× | 1.42 | Parity with FSDP within noise |
| 3 | DeepSpeed ZeRO-3 + optimizer offload | 7 | 82 | 1.35k | 4.9× | 1.42 | ~20% throughput cost for ~40% GPU mem relief |
| 4 | DeepSpeed ZeRO-3 + full offload | 4 | 130 | 0.45k | 1.6× | 1.42 | Feasibility config; ~4× throughput cost |
| 5 | DeepSpeed ZeRO-2, no offload | 22 | 34 | 1.85k | 6.7× | 1.42 | Slight throughput win at 1B; OOM expected at 7B |

- Loss parity across completed rows holds within 1%; pipeline parity confirmed.
- Memory story: FSDP `FULL_SHARD` and DeepSpeed ZeRO-3 no-offload sit within
  one bucket's worth of memory (reduce_bucket_size difference), consistent with
  the `16·N/P + activations + overhead` heuristic.
- Offload cost curve matches the chapter 3 heuristic: no cost → ~10–30% cost →
  ~2–5× cost.
```

### 2.8 Communication vs. compute shape (Part E §5, example)

From `wall_clock_breakdown` output averaged over steps 10–50 of the no-offload run:

```
[2026-11-17 14:12:03] fwd: 48.2 ms  bwd: 96.1 ms  step: 28.4 ms  comm: 22.7 ms
[2026-11-17 14:12:04] fwd: 47.9 ms  bwd: 96.5 ms  step: 28.1 ms  comm: 22.9 ms
...
```

Interpretation:

- Compute fwd + bwd ≈ 144 ms (73% of step)
- Optimizer step ≈ 28 ms (14%)
- Comm (overlapped where possible) ≈ 23 ms (12%)
- Verdict: **compute-bound-leaning with a visible communication tail**. The step is dominated by the backward pass; comm is partially overlapped but ~12% is on the critical path. Not optimizer-bound (that would show when offload is on).

Cross-check per-token FLOPs:

```
DeepSpeed FLOPS profiler: 7.42 GFLOPs/token
PaLM approximation 6·N: 6 · 1.236e9 = 7.42 GFLOPs/token
Ratio: 1.000 — agree within 1%.
```

For a vanilla decoder-only transformer with MLP intermediate size ≈ `4h` and no gradient-checkpoint recompute in this run, `6·N` is tight. If a student's profiler reports `~8·N`, they have activation checkpointing on (HFU vs. MFU; see exercise 03).

### 2.9 `report.md` (worked recommendation for the three scenarios)

```markdown
# DeepSpeed vs. FSDP recommendation report

## 1. Setup recap
Base: meta-llama/Llama-3.2-1B. Dataset: <path>, ~10k SFT rows. Hardware: 8× A100-80GB single node. Mixed precision: bf16. Seqlen: 2048. Effective batch: 128. LR schedule: cosine 1e-5. Epochs: 1. Parity with exercise 01: confirmed (final loss 1.42 in both frameworks).

## 2. Comparison table
(see §2.7)

## 3. Memory story
ZeRO-3 no-offload per-rank memory (12 GB) vs. FSDP `FULL_SHARD` (13 GB) agree within ~8%, consistent with the `16·N/P + activations + overhead` heuristic — the 1 GB delta is the different `reduce_bucket_size` default between frameworks. Both frameworks keep bf16 weights + bf16 grads + fp32 shards of optimizer state resident; neither materially beats the heuristic at steady state.

## 4. Offload cost curve
No offload → optimizer offload: 1.70k → 1.35k tok/s/GPU = 21% throughput cost. Within the 10–30% heuristic band.
Optimizer offload → full offload: 1.35k → 0.45k = 3.0× slowdown. Within the 2–5× heuristic band.

## 5. Communication vs. compute breakdown
No-offload run is compute-bound-leaning with ~12% communication tail on the critical path (wall_clock_breakdown). Backward pass is the dominant phase; comm is partially overlapped.

## 6. Recommendation

### Scenario 1 — 7–13B, 8× A100-80GB, 5M examples, iteration speed is the priority
Pick **FSDP `FULL_SHARD` throughput config** (exercise 01). Rationale: ZeRO-3 no-offload matches FSDP on memory and loss but comes ~5% behind on tok/s; the FSDP stack also has lower launcher overhead and faster save/load via `SHARDED_STATE_DICT`. Over 5M examples at ~14k tok/s, a 5% difference is a day of wall-clock. Framework switch is only warranted if a feature (ZeRO's selective fp16 weights, pipeline parallelism down the line) is on the roadmap.

### Scenario 2 — 34B base, 8× A100-80GB, 500k examples, fitting is the priority
Pick **DeepSpeed ZeRO-3 + optimizer offload**. Rationale: 34B in bf16 is 68 GB of weights; 12·N/P of FP32 AdamW state is 51 GB per rank at world size 8 — on 80 GB cards, the AdamW state will not fit alongside weights, bf16 grads, and activations without offloading it. The ~20% throughput cost is cheap insurance compared with running out of memory mid-epoch. Keep FSDP as the fallback if the host RAM budget (want >400 GB) is unavailable.

### Scenario 3 — 70B base, 4× A100-80GB, week-long budget
Pick **DeepSpeed ZeRO-3 + optimizer + parameter offload** (full offload). Rationale: 70B in bf16 is 140 GB; it will not fit on 4× 80GB (320 GB total GPU memory) once activations and gradients are added in. The 2–5× throughput hit is accepted explicitly — this is a feasibility recommendation, not a throughput recommendation. Watch host RAM carefully (need ≥ ~1 TB with pinned memory headroom); if the host cannot provide it, NVMe offload is the next lever but it compounds the throughput cost. If a 2-node 8× 80GB cluster becomes available, revisit with ZeRO-3 + optimizer offload across 8 GPUs instead.
```

### 2.10 Directory layout (as delivered)

```
outputs/exercise-02/
├── configs/
│   ├── deepspeed_accelerate.yaml
│   ├── ds_config_z3_noffload.json
│   ├── ds_config_z3_optoffload.json
│   ├── ds_config_z3_full_offload.json
│   └── ds_config_z2_noffload.json
├── train.py                       # reused from exercise 01, unchanged
├── requirements.txt               # adds deepspeed==<pin> to the ex01 set
├── results/
│   ├── deepspeed_sweep.csv
│   ├── flops_profile_noffload.txt
│   ├── wall_clock_breakdown_noffload.txt
│   └── framework_comparison.md
├── report.md
└── logs/
```

## 3. Validation steps

1. **All three configs launch.** `accelerate launch` with each of the three `DS_CONFIG_FILE` values runs at least one training step without a Trainer/DeepSpeed batch-size mismatch error. The most common failure is overriding `train_batch_size` instead of letting `"auto"` fill it — the launcher log will contain `train_batch_size != micro · accum · world`; fix by restoring `"auto"`.
2. **`zero3_init_flag: true` is honored.** Watch host RAM with `free -h` during `from_pretrained`. Without the flag, host RAM climbs by `(weight_bytes · world_size)`; with it, host RAM climbs by `weight_bytes` once (rank-0 materialization, then broadcast). On an 8-GPU node this gap is ~8× the base model size and nearly always obvious.
3. **Framework parity on loss.** Final loss across the FSDP `FULL_SHARD` row (ex01) and the DeepSpeed ZeRO-3 no-offload row (ex02) agrees within 1%. A larger gap means the recipe shifted: inspect micro-batch, accum, seqlen, LR schedule, tokenizer — in that order.
4. **Memory parity on no-offload.** Per-rank peak GPU memory for FSDP `FULL_SHARD` and DeepSpeed ZeRO-3 no-offload agree within ~15%; typical cause of larger deltas is `reduce_bucket_size` vs. `stage3_prefetch_bucket_size` defaults.
5. **Offload cost curve matches the heuristic.** `tok/s(no_offload) → tok/s(opt_offload) → tok/s(full_offload)` drops by ~10–30% then ~2–5× respectively. Outside those ranges → check host RAM contention first (`free -h` during the run), then NUMA affinity.
6. **FLOPS profiler agrees with `6·N`.** `(profiler_GFLOPs/token) / (6 · N / 1e9)` is in [0.85, 1.20]. Outside that → model has non-vanilla intermediate size or activation checkpointing is on and `8·N` applies (HFU).
7. **Host RAM headroom sanity.** `free -h` before launching an optimizer-offload run shows at least (12 bytes · N / world_size) · 1.3 free. Factor 1.3 accounts for pinned-memory overhead + data-loader + system reservations.
8. **ZeRO-2 row is documented either as a passing run or as a documented OOM.** For 1B bases it should pass with slightly higher per-rank memory than ZeRO-3; for 13B+ bases it is expected to OOM at the no-offload configuration — log "OOM as expected" with the specific allocation size and move on.
9. **Sweep CSV is complete.** One row per completed config, every column populated (GPU mem, host RAM, tok/s/GPU, tok/s total, final loss, status). OOM rows record which specific allocation failed and the next mitigation.

## 4. Rubric for grading

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Configs are correct and launch** | Any of the three configs fails to launch, or uses manual (non-`"auto"`) batch fields that conflict with Trainer | Configs launch, but `zero3_init_flag: true` missing (host OOM at 7B+) or `pin_memory` off (throughput regression) | All three configs launch cleanly, host-RAM sanity verified before run, pin_memory on |
| **Sweep coverage** | Fewer than 3 of the 3 required DeepSpeed rows populated | Three ZeRO-3 rows present but missing host-RAM column or missing final loss | Three ZeRO-3 rows + the FSDP comparison row + the ZeRO-2 row (pass or documented OOM), every column populated |
| **FLOPS profile + wall_clock_breakdown** | Not captured | Captured but not interpreted | Captured; per-token FLOPs compared to `6·N` within 20%; wall_clock_breakdown yields a compute-vs-comm-vs-step verdict |
| **Framework parity** | Loss diverges >5% between FSDP and DeepSpeed, not reconciled | Parity within 2–5%, cause suspected but not confirmed | Parity within 1%; reconciliation cause (if any delta) explicitly identified |
| **Recommendation report** | Generic ("depends on hardware"), no scenario-specific defense | Specific recommendations but defended without reference to the measured table | Each of the three scenarios gets a specific framework + offload recommendation, defended with a specific row from the comparison table |

**Pass threshold:** 7/10, with no single dimension at 0 and the recommendation report at ≥ 1.

## 5. Common mistakes

- Overriding `train_batch_size`, `train_micro_batch_size_per_gpu`, or `gradient_accumulation_steps` manually in the JSON. Trainer tries to reconcile from `TrainingArguments`, DeepSpeed refuses because the fields are already set — the job exits before step 0 with a `train_batch_size != micro · accum · world` error. Always `"auto"`.
- Dropping `zero3_init_flag: true` because "the smaller model worked without it." The smaller model worked by luck; the next base will host-OOM on multi-rank materialization.
- Running the optimizer-offload config on a RAM-contended host (data loader on the same cores, other tenants, zram enabled) and concluding "optimizer offload is slower than the heuristic predicts." It is — because the host is contended, not because the heuristic is wrong. `free -h` + `top` in a second terminal diagnose this in under a minute.
- Treating ZeRO-3's per-rank memory as "ZeRO-3 is better than FSDP." At parity (no offload), they are within ~15%; the frameworks solve the same problem with different constants. The difference is ergonomic (save/load, CLI, docs) and compatibility (PEFT + FSDP vs. PEFT + ZeRO-3), not algorithmic.
- Using the FLOPS profiler output to recompute MFU in exercise 03 without noting that activation checkpointing shifts `6·N` to `8·N` (HFU). Pick a convention and label it; see exercise 03.
- Letting `requirements.txt` drift between the FSDP (ex01) and DeepSpeed (ex02) runs. Even a `transformers` minor version bump can shift memory numbers by several percent; the cross-framework comparison is invalidated.
- Running full offload at 8 GPUs on a 128 GB host and host-OOMing partway through — then concluding "DeepSpeed is broken." It is not; the host is too small. The exercise prerequisites specify ≥256 GB for a 7B base for exactly this reason.
- Recommending "whichever framework is faster" without reference to feasibility. On a 70B base and 4× 80GB, FSDP `FULL_SHARD` simply does not fit — the recommendation's whole value is naming that constraint.
- Confusing MFU (6·N / peak) with HFU (8·N / peak) when activation checkpointing is on. Exercise 03 covers this; ex02's FLOPS cross-check is where the ambiguity usually first appears.
- Reporting throughput variance without discarding warmup steps. The first ~10 steps include CUDA graph warmup, DeepSpeed state init, and tokenizer lazy caching; a median over steps 11–60 is the honest number.

## 6. References

- DeepSpeed ZeRO documentation — stage 3, offload options, flops profiler, `wall_clock_breakdown`. <https://www.deepspeed.ai/tutorials/zero/>, <https://www.deepspeed.ai/docs/config-json/>, <https://www.deepspeed.ai/tutorials/flops-profiler/>
- Hugging Face Accelerate — DeepSpeed integration and `zero3_init_flag`. <https://huggingface.co/docs/accelerate/usage_guides/deepspeed>
- Hugging Face Transformers — Trainer × DeepSpeed integration and the `"auto"` field convention. <https://huggingface.co/docs/transformers/deepspeed>
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," 2020 (ZeRO-1/2/3 formulation). <https://arxiv.org/abs/1910.02054>
- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," 2022 (the `6·N` per-token FLOPs approximation used in this exercise and exercise 03). <https://arxiv.org/abs/2204.02311>
- NIST AI Risk Management Framework — *Manage 1.1 (risk prioritization)* and *Measure 2.1 (test methodology)* frame the "feasibility vs. throughput" distinction in the Part-E recommendation. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — *ML06 AI Supply Chain Vulnerabilities* applies to pinning `deepspeed`/`torch` versions in `requirements.txt` as supply-chain artifacts. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — ships the "framework as supply-chain component" framing used in the version-pinning validation step above. <https://atlas.mitre.org/>
