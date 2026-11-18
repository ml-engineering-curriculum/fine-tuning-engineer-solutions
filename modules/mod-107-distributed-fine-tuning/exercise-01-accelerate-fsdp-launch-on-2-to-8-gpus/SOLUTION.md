# mod-107 · exercise-01 — Accelerate + FSDP Launch on 2 → 8 GPUs — Solution

## 1. Solution overview

The deliverable is a **configuration-and-measurement artifact**, not a modeling change. The single-GPU SFT script from mod-103 is carried forward essentially unchanged; what the student produces is two `accelerate` YAML files, a six-row sweep, a per-rank sharding-verification check, and a written report that defends every FSDP knob against the memory and throughput numbers measured on the actual hardware.

Three ideas carry the exercise:

- **Sharding has to be visible in the numbers, not just in the config.** `FULL_SHARD` is only working when per-rank weight memory is near `2·N/P` bytes and per-rank optimizer state is near `12·N/P` bytes. Skipping the per-rank print is how students ship configs that silently default to `NO_SHARD` and look fine on 2 GPUs but blow up at 8.
- **Memory-conservative vs. throughput-optimized are two intentionally different operating points.** The conservative config buys headroom by giving up overlap (`BACKWARD_POST`, no forward prefetch, activation checkpointing on, batch of 1 × gradient accumulation). The throughput config spends memory for overlap and larger micro-batches (`BACKWARD_PRE`, forward prefetch on, checkpointing off if it fits). A report that produces identical-looking configs has missed the lesson.
- **Scaling has a signature.** At 2 → 4 → 8 GPUs, tok/s/GPU falls (per-rank all-gather traffic grows, batch per rank often drops). A healthy FSDP run shows ~0.6–0.9× linear scaling. If scaling is super-linear, the single-GPU baseline is underfitted (usually by activation checkpointing differences). If sub-linear beyond ~0.5×, communication is dominating and the report must name the bottleneck.

The worked configs below are written for an 8× A100-80GB single-node setup with `Llama-3.2-1B` or `Qwen2.5-1.5B`; the structure is identical for a 7B base, only the `fsdp_transformer_layer_cls_to_wrap` name and the micro-batch budget change.

## 2. Worked implementation

### 2.1 `train.py` — the SFT script, barely modified from mod-103

The single-GPU script already uses `TrainingArguments` + `SFTTrainer`. The only distributed-aware additions are per-rank logging, a post-load memory probe, and a couple of accelerator guards around saving.

```python
# train.py — identical recipe to mod-103 exercise 01 except for the rank-prefixed logger.
import os, time, json
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer

MODEL_ID = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B")
DATA_DIR = os.environ.get("SFT_DATA", "./data/sft_tokenized")

accelerator = Accelerator()
rank = accelerator.process_index
world = accelerator.num_processes

def log(msg: str) -> None:
    print(f"[rank {rank}/{world}] {msg}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
log(f"model loaded, param_count={sum(p.numel() for p in model.parameters()):,}")
log(f"post-load torch.cuda.memory_allocated={torch.cuda.memory_allocated()/1e9:.2f} GB")

dataset = load_from_disk(DATA_DIR)

args = SFTConfig(
    output_dir=f"./runs/world{world}_{os.environ.get('RUN_TAG','conservative')}",
    per_device_train_batch_size=int(os.environ.get("PDTB", "1")),
    gradient_accumulation_steps=int(os.environ.get("GAS", "16")),
    learning_rate=1e-5,
    num_train_epochs=1,
    max_steps=int(os.environ.get("MAX_STEPS", "100")),
    bf16=True,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    max_seq_length=int(os.environ.get("SEQLEN", "2048")),
    gradient_checkpointing=False,  # controlled via the accelerate config, not here
)

class StepTimer:
    def __init__(self):
        self.t_prev = time.time()
        self.records = []
    def __call__(self, step: int):
        now = time.time()
        dt = now - self.t_prev
        self.t_prev = now
        if step > 10:
            self.records.append(dt)
        log(f"step {step} step_s {dt:.3f} mem_gb {torch.cuda.max_memory_allocated()/1e9:.2f}")

timer = StepTimer()

class StepLog(torch.utils.data.IterableDataset):
    pass  # using Trainer's loop; we only need the callback

class Cb:
    def on_step_end(self, args, state, control, **kw):
        timer(state.global_step)
        return control

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=args,
    callbacks=[Cb()],
)

trainer.train()

if accelerator.is_main_process:
    median_dt = sorted(timer.records[-50:])[len(timer.records[-50:]) // 2]
    tokens_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * world * args.max_seq_length
    print(json.dumps({
        "world_size": world,
        "run_tag": os.environ.get("RUN_TAG", "conservative"),
        "median_step_s": median_dt,
        "tokens_per_step": tokens_per_step,
        "tok_per_s_total": tokens_per_step / median_dt,
        "tok_per_s_per_gpu": tokens_per_step / median_dt / world,
        "peak_mem_gb_rank0": torch.cuda.max_memory_allocated() / 1e9,
    }))
```

The script is intentionally dumb — it prints one JSON summary at the end that the sweep driver (`run_sweep.sh`) appends to `results/sweep_results.csv`.

### 2.2 `configs/fsdp_conservative.yaml` — memory first

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_processes: 8          # overridden from the launcher
mixed_precision: bf16
use_cpu: false
machine_rank: 0
main_training_function: main
rdzv_backend: static
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer   # verify with type(model.model.layers[0]).__name__
  fsdp_activation_checkpointing: true
  fsdp_backward_prefetch: BACKWARD_POST
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_use_orig_params: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_cpu_ram_efficient_loading: true
```

Companion CLI invocation (example for 8 GPUs):

```bash
PDTB=1 GAS=16 RUN_TAG=conservative \
  accelerate launch \
    --config_file configs/fsdp_conservative.yaml \
    --num_processes 8 \
    train.py
```

### 2.3 `configs/fsdp_throughput.yaml` — speed first

Only the FSDP block differs; comments mark every knob that moved.

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_processes: 8
mixed_precision: bf16
use_cpu: false
machine_rank: 0
main_training_function: main
rdzv_backend: static
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD        # keep FULL_SHARD at 7B+; drop to SHARD_GRAD_OP only if it fits per rank
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_activation_checkpointing: false      # 30% throughput back at ~4× activation memory
  fsdp_backward_prefetch: BACKWARD_PRE      # overlap all-gather of layer N-1 with bwd of layer N
  fsdp_forward_prefetch: true               # overlap next-layer all-gather with current forward
  fsdp_offload_params: false
  fsdp_use_orig_params: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_cpu_ram_efficient_loading: true
```

Companion CLI (micro-batch bisected to the highest value that fits; here 4):

```bash
PDTB=4 GAS=4 RUN_TAG=throughput \
  accelerate launch \
    --config_file configs/fsdp_throughput.yaml \
    --num_processes 8 \
    train.py
```

The effective-batch invariant `micro_batch · accum · world = 128` is held constant across both configs so loss curves are comparable.

### 2.4 Config diff table (worked example)

| Knob | conservative | throughput | Why |
|---|---|---|---|
| `fsdp_sharding_strategy` | `FULL_SHARD` | `FULL_SHARD` | Below 2× A100-80GB the model does not fit unsharded at 7B; keep FULL at 1–3B too for apples-to-apples comparison |
| `fsdp_activation_checkpointing` | `true` | `false` | ~4× activation memory for ~30% throughput; fits in throughput config because micro-batch grows and the rest of the budget absorbs it |
| `fsdp_backward_prefetch` | `BACKWARD_POST` | `BACKWARD_PRE` | `PRE` holds one extra layer's gathered params; `POST` holds none — memory for overlap |
| `fsdp_forward_prefetch` | `false` | `true` | Overlaps the next layer's all-gather with the current forward; adds a layer of gathered parameters |
| `per_device_train_batch_size` | 1 | 4 (bisected) | Larger micro-batch amortizes fixed collective cost; reduces bubble between steps |
| `gradient_accumulation_steps` | 16 | 4 | Hold effective batch (`1·16·8 = 4·4·8 = 128`) constant across configs |
| `fsdp_use_orig_params` | `true` | `true` | Mandatory for PEFT, `torch.compile`, and the stretch goals; costs nothing if you never need it |
| `fsdp_state_dict_type` | `SHARDED_STATE_DICT` | `SHARDED_STATE_DICT` | Sharded save/load is `O(N/P)`; consolidation is a separate, offline step |
| `fsdp_sync_module_states` | `true` | `true` | Materializes weights only on rank 0, broadcasts to the rest — required for `zero3_init` style workflows |
| `fsdp_cpu_ram_efficient_loading` | `true` | `true` | Streams params through CPU to avoid rank-wise peak RAM during `from_pretrained` |

### 2.5 Expected measurement table (worked numbers for an 8× A100-80GB node, Llama-3.2-1B, seqlen=2048)

These are *illustrative* reference points for calibration; a student's numbers will land in this neighborhood on comparable hardware. Treat the absolute numbers as hardware-specific — the examinable claim is the ratios and the shape of the curve, not the digits.

| Run | Config | GPUs | Peak mem/rank (GB) | Tok/s total | Tok/s/GPU | 100-step wall (s) | Final loss | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 0 | baseline (single GPU) | 1 | ~22 | ~2.2k | ~2.2k | ~730 | 1.42 | clean |
| 1 | conservative | 2 | ~12 | ~3.8k | ~1.9k | ~430 | 1.42 | clean |
| 2 | conservative | 4 | ~7 | ~7.0k | ~1.75k | ~235 | 1.42 | clean |
| 3 | conservative | 8 | ~5 | ~12.0k | ~1.5k | ~140 | 1.42 | clean |
| 4 | throughput | 2 | ~28 | ~4.6k | ~2.3k | ~355 | 1.42 | clean |
| 5 | throughput | 4 | ~18 | ~8.4k | ~2.1k | ~195 | 1.42 | clean |
| 6 | throughput | 8 | ~13 | ~14.5k | ~1.8k | ~113 | 1.42 | clean |

**Reading the table**:

- Tok/s/GPU drops as world size grows (all-gather traffic rises); 8-GPU throughput run at `~1.8 × 8 / 2.2 ≈ 6.5×` of baseline tok/s is healthy.
- Throughput config spends ~3× the per-rank memory of conservative for ~20% more tok/s/GPU. Worth it on 80GB cards; not worth it on 40GB cards where OOM headroom is the binding constraint.
- Loss is identical across runs (within mixed-precision noise). Divergent loss means the effective batch or LR schedule shifted, not the sharding.

### 2.6 `verification.md` — proving sharding actually happened

```markdown
# Sharding verification (8 GPUs, FULL_SHARD, throughput config)

Base model: meta-llama/Llama-3.2-1B
Param count N = 1,235,814,400
Per-rank expected weight memory (bf16) ≈ 2·N/P = 2 · 1.236e9 / 8 ≈ 309 MB
Per-rank expected AdamW state (fp32) ≈ 12·N/P = 12 · 1.236e9 / 8 ≈ 1.85 GB

## Post-load weight memory (measured)
[rank 0/8] post-load torch.cuda.memory_allocated=0.34 GB
[rank 3/8] post-load torch.cuda.memory_allocated=0.34 GB
[rank 7/8] post-load torch.cuda.memory_allocated=0.34 GB

0.34 GB ≈ 2·N/P. If it were ~2.5 GB, sharding is NOT active and the config is
defaulting to NO_SHARD — the first thing to check is
`fsdp_transformer_layer_cls_to_wrap` (must match the actual layer class name
printed by `type(model.model.layers[0]).__name__`).

## Post-step optimizer state (after first optimizer.step)
[rank 0/8] optim state sz 1.89 GB
[rank 3/8] optim state sz 1.89 GB

Measured via:
    sum(p.element_size() * p.numel()
        for g in optimizer.state.values()
        for p in g.values() if torch.is_tensor(p))

## Sharded checkpoint file count
accelerator.save_state("test_ckpt/")
ls test_ckpt/
> model.safetensors_shard_rank0_of_8 ... model.safetensors_shard_rank7_of_8
> optimizer.bin_shard_rank0_of_8 ... (8 files)
8 shards, matches world_size — sharded save is correct.

## Activation checkpointing (throughput config has it off; conservative has it on)
Profiler trace of conservative config contains `CheckpointFunctionBackward` in
the backward pass. Trace of throughput config does not. Expected.
```

### 2.7 `report.md` (skeleton the student fills in with their own numbers)

```markdown
# FSDP sweep report

## 1. Baseline recap
Single-GPU: <model>, seqlen <s>, eff batch 128, 100 steps, peak 22 GB, 2.2k tok/s, loss 1.42.

## 2. Config diff
(see table in §2.4)

## 3. Results
(see table in §2.5)
- Scaling efficiency at 8 GPUs, throughput config:
  8 · tok/s/GPU_1 ≈ 17.6k nominal; measured 14.5k → 82% efficiency.
- Conservative loses more at scale: 12.0k / (8 · 2.2k) ≈ 68%.
- Headline: throughput config gives 20% better per-GPU tok/s than conservative
  at 8 GPUs; conservative gives ~2.6× the memory headroom.

## 4. Sharding verification
Per-rank post-load weights 0.34 GB ≈ 2·N/P = 0.31 GB. Optimizer state 1.89 GB ≈
12·N/P = 1.85 GB. Sharded checkpoint produces 8 files. FULL_SHARD is live.

## 5. Bottleneck analysis
8-GPU throughput run: `nvidia-smi dmon -s pucm` shows GPU util oscillating
88-100% with a visible dip during backward. Profiler trace shows
`ncclKernel_ReduceScatter` on the backward critical path, 11% of step time not
overlapped. This is a communication-bound shape — levers would be
`fsdp_backward_prefetch: BACKWARD_PRE` (already on), larger micro-batch, or
HYBRID_SHARD on multi-node.

## 6. Recommendation
For a real production SFT job at 7B on this cluster: start with the throughput
config. Swap to the conservative config only if (a) the model is ≥ 13B and
80 GB is the hard memory ceiling, (b) sequence length must exceed 4096 for the
task, or (c) the team prioritizes repeatability over tok/s. Both configs
produce identical loss curves within 1% at the same effective batch; the choice
is about hardware headroom, not about the model.
```

### 2.8 `requirements.txt` (worked pin, update per the current compatibility matrix for your `torch` version)

```text
torch==2.4.1
accelerate==1.0.1
transformers==4.45.2
datasets==3.0.1
trl==0.11.3
peft==0.13.1
flash-attn==2.6.3        # install last; requires matching torch + cuda
sentencepiece==0.2.0
bitsandbytes==0.44.0     # kept for the LoRA + QLoRA stretch goals
```

Re-validate the FSDP2 + Accelerate + TRL compatibility matrix against the current release notes at install time; the pins above are known-good on the reference node but drift quickly.

### 2.9 Directory layout (as delivered)

```
outputs/exercise-01/
├── configs/
│   ├── fsdp_conservative.yaml
│   └── fsdp_throughput.yaml
├── train.py
├── requirements.txt
├── baseline_metrics.md
├── results/
│   └── sweep_results.csv
├── verification.md
├── report.md
└── logs/                 # per-rank stdout for the 8-GPU runs at minimum
```

## 3. Validation steps

1. **Baseline first.** Run `train.py` once on 1 GPU and record `baseline_metrics.md`. Everything downstream is measured against it.
2. **Launch both configs on 2 GPUs.** Both should complete 100 steps cleanly with the effective batch invariant held constant. Loss at step 100 must match the single-GPU baseline within ±1%; a divergence means the effective batch or LR schedule drifted.
3. **Per-rank sharded memory check (the exercise's most-skipped step).** After the first optimizer step, log `torch.cuda.max_memory_allocated()` on at least two ranks. Weight memory must be close to `2·N/P` and optimizer state close to `12·N/P`. If weight memory is close to `2·N`, FULL_SHARD is not actually active — usually because `fsdp_transformer_layer_cls_to_wrap` does not match the model's decoder layer class name.
4. **Sharded save/load round-trip.** `accelerator.save_state("ckpt/")` produces `world_size` files. Relaunch at the same world size with `accelerator.load_state("ckpt/")` and verify training resumes with no loss spike on the next step.
5. **FA2 is live.** `attn_implementation="flash_attention_2"` is in `from_pretrained`, and the PyTorch profiler (exercise 03) shows `flash_fwd_*` and `flash_bwd_*` kernels. Without FA2, activation memory is `O(batch · heads · seq²)` and the exercise's memory predictions do not line up with the measured numbers.
6. **Sweep CSV is complete.** `wc -l results/sweep_results.csv` is 7 (1 header + 6 runs). Every row has `peak_mem_gb`, `tok_per_s_per_gpu`, `median_step_s`, `final_loss`, and `status ∈ {clean, oom, hang}`. OOM at a known micro-batch counts as a real result.
7. **Scaling efficiency.** At 8 GPUs the throughput config's tok/s should be meaningfully greater than `4 × single-GPU tok/s`. `world_size · 0.6` is a floor; below it, the report must name the bottleneck (usually communication, which the trace confirms).
8. **Config diff defensibility.** For every knob in `fsdp_conservative.yaml` that differs from `fsdp_throughput.yaml`, the report's §2 table has a one-line justification. Missing justifications are a hard fail.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Configs are correct and distinct** | Missing one config; or both configs are identical | Both configs launch, but conservative is not actually memory-conservative (e.g., `BACKWARD_PRE` on, checkpointing off) | Both configs launch; every knob that differs is defensible and the diff table in the report names each |
| **Sweep coverage** | Fewer than 4 of 6 runs populated | All 6 runs recorded, but some fields missing (e.g., no per-rank memory, no final loss) | All 6 runs complete with every Part-D field, OOM rows explicitly documented with next-mitigation |
| **Sharding verification** | Not attempted | Per-rank memory logged but not compared to the `2·N/P` expectation | Per-rank weight memory ≈ `2·N/P`, optimizer state ≈ `12·N/P`, sharded checkpoint produces `world_size` files, activation-checkpointing presence confirmed in the trace |
| **Scaling analysis** | Numbers reported with no interpretation | Scaling efficiency computed but bottleneck not named | Scaling efficiency reported, bottleneck named with specific evidence (`nvidia-smi dmon` or profiler-trace excerpt) |
| **Report is a document a reviewer can act on** | Missing sections, mostly code dumps | All six Part-F sections present but recommendation is generic ("it depends") | All six sections, recommendation names a specific config for a specific scenario and defends with a specific measurement from the sweep |

**Pass threshold:** 7/10, with no single dimension at 0 and the sharding verification at 2.

## 5. Common mistakes

- Treating `fsdp_transformer_layer_cls_to_wrap` as cosmetic. If it does not match the actual decoder-layer class name, Accelerate falls back to a different auto-wrap policy (usually size-based) and the sharding factor silently shifts. Always print `type(model.model.layers[0]).__name__` before authoring the config.
- Bisecting `per_device_train_batch_size` on the *conservative* config, not the throughput one. Conservative is defined by low memory; its batch is 1 by design. Throughput is where you binary-search the largest batch that fits without OOM.
- Not holding effective batch constant across configs. `micro · accum · world = const` is the invariant that keeps loss curves comparable. Without it, "the throughput config trains to a different loss" is a recipe bug, not a sharding insight.
- Using the FP32 peak-FLOPS number for scaling-efficiency math. For bf16 training on Ampere/Hopper you compare against the bf16 tensor-core peak; see exercise 03.
- Shipping configs without a sharded-checkpoint save/load test. The first time a student needs to resume a training run is also the first time they discover the `fsdp_state_dict_type` was wrong, and the shards they wrote cannot be read back.
- Running the sweep with activation checkpointing differently flipped between configs and world sizes. Pick one invariant — "both configs use checkpointing" or "conservative uses it, throughput does not" — and keep it. Flipping ad-hoc at the 8-GPU run poisons the comparison.
- Logging rank-0-only memory. Rank 0 is slightly heavier (CPU-side bookkeeping) than other ranks on FSDP; always log at least rank 0 and one non-zero rank to catch rank-heterogeneous bugs (`fsdp_sync_module_states: false` is a classic cause).
- Forgetting to pin versions. `accelerate`, `transformers`, `trl`, and `torch` interact closely; a minor-version drift mid-exercise invalidates all six runs in the sweep.
- Claiming "it OOM'd" without a next-mitigation. An OOM at step 0 on the throughput-config 2-GPU run tells you to lower micro-batch to 2 and rerun, or turn checkpointing on. The sweep CSV row should record `status=oom` *and* `next_mitigation=micro_batch_2`.
- Using the throughput config on a tight-memory card (e.g., 24 GB consumer GPU) and concluding "FSDP is broken." The exercise is written for 40 GB / 80 GB SKUs; smaller cards need the conservative config as the *baseline*, not the comparison.

## 6. References

- Hugging Face Accelerate — FSDP integration docs. <https://huggingface.co/docs/accelerate/usage_guides/fsdp>
- PyTorch FSDP (Fully Sharded Data Parallel) tutorial and API reference (FSDP1); `fully_shard` (FSDP2) API. <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>, <https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>
- Hugging Face Transformers — `attn_implementation="flash_attention_2"` and the `TrainingArguments` fields referenced above. <https://huggingface.co/docs/transformers/perf_train_gpu_one>
- NIST AI Risk Management Framework — *Measure 4.2 (operational monitoring)* and *Manage 1.1 (risk prioritization)* apply to the per-rank memory and loss-parity checks that gate releasing a training config. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — *ML06 AI Supply Chain Vulnerabilities* is the framing for pinning `requirements.txt` against a reproducible stack; distributed-training configs are supply-chain artifacts. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — tactics catalog; the sharded-checkpoint integrity story (verifying all `world_size` shards present after save) is a hygiene check against supply-chain and model-integrity tactics. <https://atlas.mitre.org/>
- TRL `SFTTrainer` reference used for the training loop. <https://huggingface.co/docs/trl/sft_trainer>
