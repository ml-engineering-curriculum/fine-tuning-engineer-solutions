# mod-107 · exercise-04 — OOM and NCCL Debugging Drills — Solution

## 1. Solution overview

The deliverable is five per-scenario directories (`scenario-01` through `scenario-05`), each containing `injection.diff`, `log.txt`, `diagnosis.md`, `reproducer.py`, and `fix.md`, plus a one-page `summary.md` with a cross-scenario table, an honest self-assessment, and escalation notes. The exercise is scored on the **diagnostic reflex** — how directly a student maps log evidence to a failure family — not on the fix, which is usually obvious once the family is named.

Three ideas carry the exercise:

- **A reproducer is the artifact, not an afterthought.** A good reproducer is 20–50 lines, uses `torch.distributed` directly (not `accelerate` or `SFTTrainer`), and exhibits the failure in under 10 seconds. "I re-ran the training script" is not a reproducer; it is a failed training run. The reproducer is what a student would attach to an issue filed against PyTorch, DeepSpeed, or a platform team.
- **Diagnostic env vars are muscle memory.** `NCCL_DEBUG=INFO`, `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180` must be set in the launch command for every scenario. Students who launch with defaults get a 30-minute watchdog timeout and a log stripped of the detail needed to diagnose. Practice setting them so the muscle memory is there when a 16-GPU run hangs on Friday.
- **Signatures, not pattern-matching.** Each failure family has a distinct log signature — OOM at step 0 vs. drifting memory over 200 steps; a silent hang where `nvidia-smi` shows processes resident and GPU util at 0; a collective timeout where all surviving ranks log the same timeout except the one that exited. The exercise builds the reflex to read the signature first and only then apply the fix.

The worked content below follows the four-step protocol (reproduce → diagnose → minimal reproducer → fix) for each scenario, then a cross-scenario summary.

## 2. Worked implementation

### 2.1 Launch wrapper — always set the diagnostic env vars

```bash
# scripts/launch_debug.sh
#!/usr/bin/env bash
set -euo pipefail

NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch \
    --config_file "${ACCEL_CFG:-configs/fsdp_throughput.yaml}" \
    --num_processes "${NPROC:-4}" \
    train.py "$@" 2>&1 | tee -a "${LOG_FILE:-logs/run.log}"
```

Scenarios override specific env vars (e.g., scenario 3 sets `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180`). The wrapper otherwise stays constant across all five scenarios so log formats are directly comparable.

### 2.2 Scenario 1 — OOM at step 0

**Reproduce.** Two interchangeable injection paths; the solution ships the FA2-off version because the log evidence is cleaner.

```diff
# scenario-01/injection.diff
--- a/train.py
+++ b/train.py
 model = AutoModelForCausalLM.from_pretrained(
     MODEL_ID,
     torch_dtype=torch.bfloat16,
-    attn_implementation="flash_attention_2",
+    # attn_implementation="flash_attention_2",   # intentionally off for the OOM drill
 )

--- a/configs/fsdp_throughput.yaml
+++ b/configs/fsdp_throughput.yaml
-per_device_train_batch_size: 4
+per_device_train_batch_size: 4
 # separately, bump seqlen in the launcher: SEQLEN=4096
```

**Log excerpt (`scenario-01/log.txt`).**

```
[rank 0/4] model loaded, param_count=1,235,814,400
[rank 0/4] step 0 forward begin
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.50 GiB
(GPU 0; 79.14 GiB total capacity; 68.40 GiB already allocated; 1.21 GiB free;
68.52 GiB reserved in total by PyTorch) If reserved memory is >> allocated
memory try setting max_split_size_mb to avoid fragmentation.
```

**Diagnose (`scenario-01/diagnosis.md`).**

```markdown
## Signature: OOM at step 0 inside the first forward pass

Failure family: activation memory spike (chapter 4 memory component: activations).
Specifically: the `[b, a, s, s]` attention-score tensor is being materialized
because FlashAttention is off. For seqlen 4096, 32 heads, bf16, micro-batch 4:
    activation_size = 4 (b) · 32 (heads) · 4096 (s) · 4096 (s) · 2 (bf16)
                   = 4.3 GB per layer
    stacked across 32 layers = ~140 GB — far exceeds per-rank budget after
    FULL_SHARD weight/grad/optimizer allocations.

Evidence from the log:
- "Tried to allocate 12.50 GiB" — single allocation, not a growing total.
- "68.40 GiB already allocated" — most of card already consumed by the time
  this allocation lands.
- Fails at step 0 during forward, not during backward or optimizer step.
- The preceding profiler trace (ex03) showed FA2 kernels; running the same
  config without FA2 removes them and materializes the explicit attention
  matrix — this is the proximate cause.

NOT a drift, NOT a hang, NOT a collective issue. Pure allocation failure.
```

**Minimal reproducer (`scenario-01/reproducer.py`).**

```python
# scenario-01/reproducer.py — single-GPU, no FSDP, no Trainer.
# Reproduces the attention-score OOM in ~20 lines.
import torch

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

B, H, S = 4, 32, 4096    # batch, heads, seq
D = 128                  # head dim
dtype = torch.bfloat16
dev = "cuda"

q = torch.randn(B, H, S, D, dtype=dtype, device=dev, requires_grad=True)
k = torch.randn(B, H, S, D, dtype=dtype, device=dev, requires_grad=True)
v = torch.randn(B, H, S, D, dtype=dtype, device=dev, requires_grad=True)

# Plain scaled-dot-product (what you get without FA2): materializes [B, H, S, S].
scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)    # <-- the 4.3 GB tensor
probs = scores.softmax(dim=-1)
out = probs @ v

loss = out.sum()
loss.backward()
print(f"peak_mem_gb={torch.cuda.max_memory_allocated()/1e9:.2f}")
```

Running on a single 80GB card with `B=4, H=32, S=4096` prints `peak_mem_gb≈18.8` *without* FSDP overhead; wrap in FSDP + 32 layers and the real training config OOMs. The reproducer isolates the attention-score tensor as the dominant allocation.

**Fix (`scenario-01/fix.md`).**

```markdown
## Fix
1. Re-enable `attn_implementation="flash_attention_2"` in `from_pretrained`.
2. Keep seqlen 4096, micro-batch 4. The job now runs.

Verification:
- Peak per-rank memory dropped from OOM (>80 GB) to 42 GB.
- Step time at seqlen 4096 settled at ~0.21 s/step (vs. ~0.11 s/step at
  seqlen 2048 — ~2× cost, as expected for attention compute scaling in FA2
  which is O(s²) in compute but O(s) in memory).

## Secondary mitigations if FA2 were truly unavailable
- Turn `fsdp_activation_checkpointing: true` (the exercise 01 conservative
  config's choice). Buys ~3–4× activation memory for a ~30% throughput cost.
- Reduce seqlen to 2048 (if the dataset allows).
- Reduce micro-batch from 4 to 2.

## Tiered mitigation order (the ordered fallbacks menu)
FA2 on → checkpointing on → micro-batch halve → seqlen reduce → 8-bit optimizer → optimizer offload.
Apply in order; stop at the first that fits.
```

### 2.3 Scenario 2 — OOM at step ~200 (drifting memory)

**Reproduce.** The injection keeps a reference to the last hidden state per step.

```diff
# scenario-02/injection.diff
--- a/train.py
+++ b/train.py
 class Cb:
+    _debug_history = []    # BUG: retains intermediate tensors forever
     def on_step_end(self, args, state, control, **kw):
+        outputs = kw.get("outputs")
+        if outputs is not None and hasattr(outputs, "hidden_states"):
+            self._debug_history.append(outputs.hidden_states[-1])   # <-- leak
         timer(state.global_step)
         return control
```

(If the student's training loop doesn't surface `outputs` in the callback, they can inject the leak directly inside the `training_step` override. The specific hook matters less than the pattern.)

**Log excerpt (`scenario-02/log.txt`).** Per-step memory on rank 0 grows ~70 MB per step; the rest of the ranks stay flat.

```
[rank 0/4] step 10  mem_gb 14.21
[rank 0/4] step 50  mem_gb 16.95
[rank 0/4] step 100 mem_gb 20.78
[rank 0/4] step 150 mem_gb 24.60
[rank 0/4] step 200 mem_gb 28.45
[rank 0/4] step 210 mem_gb 29.21
[rank 0/4] step 218 torch.OutOfMemoryError: CUDA out of memory.
[rank 1/4] step 218 mem_gb 14.05   # <-- other ranks stayed flat
[rank 2/4] step 218 mem_gb 14.05
[rank 3/4] step 218 mem_gb 14.05
...
[rank 1/4] [NCCL] Watchdog caught collective operation timeout...   # after ~3 min
```

**Diagnose (`scenario-02/diagnosis.md`).**

```markdown
## Signature: monotonically growing memory on one rank, eventual OOM

Failure family: Python-side memory leak (reference retention, not framework bug).
Specifically: the `_debug_history` list on rank 0's callback holds references
to the last hidden state of every step. Each hidden state is
[batch=4, seq=2048, hidden=2048] in bf16 = 67 MB; 200 steps = ~13 GB of leaked
state on rank 0 only. Explains both the drift and the rank-asymmetry.

Evidence:
- Per-step memory line shows monotonic growth of ~70 MB/step on rank 0.
- Other ranks stay flat at ~14 GB. This is the critical piece: FSDP memory
  distribution is symmetric; growth on only one rank cannot be sharding-side.
  Must be a Python-side retention.
- After rank 0 OOMs, the remaining ranks hit the next collective and time out
  (expected; NOT a separate failure).

Localization:
- `torch.cuda.memory._record_memory_history(enabled=True, max_entries=100_000)`
  before the training loop, then `torch.cuda.memory._dump_snapshot("mem.pickle")`
  on OOM. Load the pickle at <pytorch.org/memory_viz>.
- The viewer shows a growing stack of hidden-state allocations attributed to
  the `training_step` → `on_step_end` → `_debug_history.append` callstack.
- Screenshot: scenario-02/memory_viz_screenshot.png
```

**Minimal reproducer (`scenario-02/reproducer.py`).**

```python
# scenario-02/reproducer.py — isolates the Python-side retention bug.
import torch

history = []    # the leak
device = "cuda"

for step in range(200):
    x = torch.randn(4, 2048, 2048, dtype=torch.bfloat16, device=device)
    y = x.tanh()
    loss = y.sum()
    loss.backward()
    history.append(y.detach())   # <-- retention
    if step % 20 == 0:
        print(f"step {step} mem_gb {torch.cuda.memory_allocated()/1e9:.2f}")
```

Outputs a clean monotonic growth; drop the `history.append(...)` line and memory stays flat.

**Fix (`scenario-02/fix.md`).**

```markdown
## Fix
Remove the `_debug_history.append(...)` line. If the intermediate is genuinely
needed for logging:

```python
# Correct pattern: move to CPU and stream to disk.
with open("hidden_states.bin", "ab") as f:
    f.write(outputs.hidden_states[-1].detach().to("cpu", dtype=torch.float16).numpy().tobytes())
```

Verification: per-step rank-0 memory lines flat at ~14 GB from step 10 to step
500 (last check). No OOM over 500 steps.

## Secondary lesson
If the leak must be allowed to exist (e.g., for a one-off debugging run),
`torch.no_grad()` around the retained tensor is not enough — it blocks the
gradient graph but not the forward tensor. `.detach().cpu()` is the pair that
prevents both graph retention and GPU retention.
```

### 2.4 Scenario 3 — NCCL collective timeout

**Reproduce.** Exit one rank intentionally and set a short heartbeat timeout.

```diff
# scenario-03/injection.diff
--- a/train.py
+++ b/train.py
 for step, batch in enumerate(loader):
+    if step == 5 and accelerator.process_index == accelerator.num_processes - 1:
+        print(f"[rank {accelerator.process_index}] intentionally exiting early", flush=True)
+        import sys; sys.exit(0)
     outputs = model(**batch)
     ...
```

Launch with `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180`.

**Log excerpt (`scenario-03/log.txt`).**

```
[rank 0/4] step 4 loss 1.59 mem_gb 14.02
[rank 1/4] step 4 loss 1.59 mem_gb 14.02
[rank 2/4] step 4 loss 1.59 mem_gb 14.02
[rank 3/4] step 4 loss 1.59 mem_gb 14.02
[rank 3/4] intentionally exiting early
[rank 0/4] step 5 loss 1.58 mem_gb 14.02
[rank 1/4] step 5 loss 1.58 mem_gb 14.02
[rank 2/4] step 5 loss 1.58 mem_gb 14.02
[rank 0/4] step 6 backward begin
[rank 1/4] step 6 backward begin
[rank 2/4] step 6 backward begin
NCCL INFO Channel 00 : 0[0] -> 3[3] via NET/IB/0 — unreachable
NCCL INFO AllGather: timeout detected on rank 0, waiting for peers
...
[rank 0/4] [NCCL] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=217, OpType=ALLGATHER, Timeout(ms)=180000)
[rank 0/4] aborted
[rank 1/4] [NCCL] Watchdog caught collective operation timeout...
[rank 2/4] [NCCL] Watchdog caught collective operation timeout...
```

**Diagnose (`scenario-03/diagnosis.md`).**

```markdown
## Signature: all surviving ranks time out on the same collective at the same step

Failure family: collective-participant disappearance ("peer went away silently").

Evidence:
- Ranks 0–2 log steps normally through step 5; rank 3 logs the intentional
  exit; then ranks 0–2 enter step 6 and hang on AllGather for exactly 180 s
  (the `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` we set).
- Each surviving rank's watchdog logs the same OpType (ALLGATHER) and same
  SeqNum. Rank identity of the *missing* rank is inferred from the one rank
  whose log stops cold with no error: that's your culprit.
- With `NCCL_DEBUG=INFO`, we also see `0[0] -> 3[3] via NET/IB/0 — unreachable`,
  which pinpoints rank 3 before the watchdog fires.

In the wild, the silent-exit cause has many origins: segfault in rank 3's
process, Xid 79 (GPU fallen off bus), OOM killer (`dmesg | grep oom-killer`),
spot preemption, a cluster manager terminating the rank for exceeding a quota.
The diagnostic habit is the same: identify the silent rank first, then look at
its last stdout line + `dmesg` + `nvidia-smi` for context.

Why `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180` matters: without it, the watchdog
default is ~30 minutes. Students who don't set this wait 30 minutes to see the
timeout. The exercise's point is to feel the loop.
```

**Minimal reproducer (`scenario-03/reproducer.py`).**

```python
# scenario-03/reproducer.py — two ranks, one exits mid-allreduce.
# Launch: torchrun --nproc_per_node 2 reproducer.py
import os, sys, torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)

for step in range(10):
    t = torch.ones(16_000_000, device="cuda") * (rank + 1)
    if step == 2 and rank == 1:
        print(f"[rank 1] exiting", flush=True)
        sys.exit(0)
    dist.all_reduce(t)
    print(f"[rank {rank}] step {step} ok sum={t.sum().item():.0f}", flush=True)
```

Launch with `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60 torchrun --nproc_per_node 2 reproducer.py`. Rank 0 hangs on the step-2 all_reduce and watchdog-aborts after 60 s.

**Fix (`scenario-03/fix.md`).**

```markdown
## Fix
The exiting rank must not exit, OR all ranks must exit together.

## General pattern
Any code path that can early-return must either:
- be unreachable on a rank-dependent branch (don't have per-rank early returns);
- have a `dist.barrier()` before the return so the exiting rank participates
  in one last collective and all ranks know to stop; or
- replace the early-return with a per-rank `should_stop: bool` that gets
  `dist.all_reduce`'d so every rank observes the same stop condition.

## Pattern code (what the training loop should look like)
```python
should_stop_local = torch.tensor([int(local_cond)], device="cuda")
dist.all_reduce(should_stop_local, op=dist.ReduceOp.MAX)
if should_stop_local.item():
    break   # every rank breaks together
```

## Verification
After removing the per-rank early exit and replacing with the all_reduce
pattern, the training loop completes 100 steps cleanly on all 4 ranks with no
timeout.
```

### 2.5 Scenario 4 — Silent hang mid-run

**Reproduce.** Only two of four ranks hit a barrier.

```diff
# scenario-04/injection.diff
--- a/train.py
+++ b/train.py
 for step, batch in enumerate(loader):
+    if step == 50 and accelerator.process_index < 2:
+        accelerator.wait_for_everyone()   # only ranks 0 and 1 reach this
     outputs = model(**batch)
     ...
```

No short heartbeat timeout set — the job hangs indefinitely.

**Log excerpt (`scenario-04/log.txt`).**

```
[rank 0/4] step 49 loss 1.44 mem_gb 14.02
[rank 1/4] step 49 loss 1.44 mem_gb 14.02
[rank 2/4] step 49 loss 1.44 mem_gb 14.02
[rank 3/4] step 49 loss 1.44 mem_gb 14.02
[rank 0/4] step 50 begin
[rank 1/4] step 50 begin
[rank 2/4] step 50 begin
[rank 3/4] step 50 begin
# --- ONE MINUTE OF SILENCE ---
```

**`scenario-04/nvidia-smi.txt`** (sampled 90 s into the hang):

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   PID       Process name                            GPU Memory Usage   |
|  0     12345     python train.py                          14.1 GiB          |
|  1     12346     python train.py                          14.1 GiB          |
|  2     12347     python train.py                          14.1 GiB          |
|  3     12348     python train.py                          14.1 GiB          |
+-----------------------------------------------------------------------------+
| GPU-Util %  0  1  2  3                                                      |
|              0  0  0  0  <-- all four at 0, but processes still resident    |
+-----------------------------------------------------------------------------+
```

**`scenario-04/pyspy_dump.txt`** (collected simultaneously with `for pid in $(pgrep -f train.py); do echo "=== $pid ==="; py-spy dump --pid $pid; done`):

```
=== PID 12345 (rank 0) ===
Thread 12345: active
  _distributed_c10d.cpp:barrier (in torch._C._distributed_c10d)
  distributed_c10d.py:barrier
  accelerator.py:wait_for_everyone
  train.py:52   <-- in our injected barrier

=== PID 12346 (rank 1) ===
Thread 12346: active
  _distributed_c10d.cpp:barrier
  accelerator.py:wait_for_everyone
  train.py:52   <-- in our injected barrier

=== PID 12347 (rank 2) ===
Thread 12347: active
  nn/functional.py:scaled_dot_product_attention
  modeling_llama.py:forward
  train.py:56   <-- in the normal training step

=== PID 12348 (rank 3) ===
Thread 12348: active
  nn/functional.py:scaled_dot_product_attention
  modeling_llama.py:forward
  train.py:56   <-- in the normal training step
```

**Diagnose (`scenario-04/diagnosis.md`).**

```markdown
## Signature: no error, no progress, processes resident with GPU util 0

Failure family: code-path divergence on a collective (some ranks called, others
did not).

Evidence (the single piece that pins it):
- py-spy dumps show ranks 0 and 1 blocked in `wait_for_everyone` → `barrier`.
- py-spy dumps show ranks 2 and 3 in the normal forward pass at
  `scaled_dot_product_attention`.
- Collectives are contracts: every rank must enter, or none does. Half the
  ranks are in the barrier; the other half are still training. The barrier
  will never complete.

Why `nvidia-smi` doesn't tell the story: processes are resident, memory is
allocated, GPUs are idle (no kernels running). That's exactly what a code-path
divergence looks like — the processes are alive, they are just not making
progress and never will.

Why the watchdog didn't fire: no short heartbeat set. The default watchdog
timeout is long enough that a student can wait tens of minutes without the
error surfacing. This is why `py-spy` is the right tool, not "wait for the
error" — the error is minutes-to-hours away.

py-spy one-liner (from the guidance):
    for pid in $(pgrep -f "train.py"); do
      echo "=== PID $pid ===";
      py-spy dump --pid $pid;
    done
```

**Minimal reproducer (`scenario-04/reproducer.py`).**

```python
# scenario-04/reproducer.py — two ranks, one calls barrier, one does not.
# Launch: torchrun --nproc_per_node 2 reproducer.py
import time, torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

if rank == 0:
    print("[rank 0] about to call barrier", flush=True)
    dist.barrier()
    print("[rank 0] past barrier (should never print)", flush=True)
else:
    print("[rank 1] sleeping, NOT calling barrier", flush=True)
    time.sleep(120)
```

Running this and `py-spy dump`'ing both processes within the first 5 seconds reproduces the signature: rank 0 in barrier, rank 1 sleeping.

**Fix (`scenario-04/fix.md`).**

```markdown
## Fix
Remove the conditional collective. The general principle:

> Collectives are contracts. Every rank participates, or none does.

If a logging / checkpoint / eval step only needs to run on rank 0, do the
actual work on rank 0 after a barrier that all ranks hit:

```python
# All ranks enter the barrier.
accelerator.wait_for_everyone()
# Rank 0 does the one-sided work.
if accelerator.is_main_process:
    save_checkpoint(...)
# All ranks enter the next barrier to resume together.
accelerator.wait_for_everyone()
```

## Verification
After removing the conditional barrier, the training loop completes 100 steps
on all 4 ranks with no hang.

## Secondary lesson
If the training job starts to hang silently in production, the first move
is always: `py-spy dump` every rank. If ranks are in different Python
stacks at the same wall-clock moment, you have code-path divergence.
```

### 2.6 Scenario 5 — Load imbalance under variable-length sequences

**Reproduce.** Disable length-bucketed sampling and sequence packing.

```diff
# scenario-05/injection.diff
--- a/train.py
+++ b/train.py
-sampler = LengthGroupedSampler(
-    dataset["train"], batch_size=args.per_device_train_batch_size, lengths=lengths,
-)
+sampler = torch.utils.data.RandomSampler(dataset["train"])   # <-- no bucketing

 def collate_fn(batch):
-    # previously: pack to max-seqlen with proper attention masks
-    return pack_and_pad(batch, max_seq_len=args.max_seq_length)
+    # naive collate: pad each batch to its longest member
+    return naive_pad_to_longest(batch)
```

**Log excerpt (`scenario-05/log.txt`), per-step wall and max seqlen.**

```
step 0  step_s 3.1  batch_max_seqlen 480
step 1  step_s 11.8 batch_max_seqlen 3996
step 2  step_s 4.2  batch_max_seqlen 672
step 3  step_s 11.5 batch_max_seqlen 4001
step 4  step_s 3.8  batch_max_seqlen 540
...
P50 step_s 6.4; P90 step_s 11.6; min 3.1; max 12.1
```

**`scenario-05/nvidia-smi-dmon.log`** (60 s of `nvidia-smi dmon -s pucm -d 1`) shows alternating full-utilization and 20–40% utilization windows, with the low windows corresponding to the short-batch steps (some ranks finishing early and idling at the collective while the longest-seqlen rank catches up).

**Diagnose (`scenario-05/diagnosis.md`).**

```markdown
## Signature: wildly variable step times, uneven per-GPU utilization, loss normal

Failure family: load imbalance (variable-length sequences under FSDP/ZeRO-3).

Evidence:
- Per-step wall time varies 3–4× (3.1 s to 12.1 s). Baseline exercise-01
  throughput run had 5% variance; this run has 100%+ variance.
- Step time correlates strongly with `batch_max_seqlen`:
    short batches (max 400–600) → 3–4 s
    long batches (max 3900–4100) → 11–12 s
  Ratio ~3–4×. Attention compute under FA2 is O(s²) in compute, so the
  jump from seqlen 500 to 4000 should be ~64× in attention compute alone;
  the fact that overall step time rises "only" 3–4× indicates attention is
  NOT the step's full cost, consistent with a decent MFU.
- `dmon` output shows ranks finishing their forward/backward at different
  wall-clock times depending on their batch; FSDP's all-reduce/reduce-scatter
  waits for the slowest rank. The rank with the longest sequence is the
  critical path for every step it participates in.

Why FSDP / ZeRO-3 does not help this: activation memory does NOT shard across
DP ranks. Each rank materializes its own batch's activations in full. A rank
whose batch happens to contain a 4000-token sequence does 8× the attention
work of a rank whose batch is 500 tokens; the collectives at the end of the
step wait for the slow one. This is a data-pipeline issue, not a sharding issue.
```

**Minimal reproducer (`scenario-05/reproducer.py`).**

```python
# scenario-05/reproducer.py — one process, shows step time vs. max seq length.
# Not multi-rank; the point is the per-step cost model, which the imbalance amplifies.
import time, random, torch
import torch.nn as nn

torch.manual_seed(0)
dev = "cuda"
H, D = 32, 128   # heads, head_dim

class MiniBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.wqkv = nn.Linear(H*D, 3*H*D, bias=False).to(dev, torch.bfloat16)
        self.wo = nn.Linear(H*D, H*D, bias=False).to(dev, torch.bfloat16)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.wqkv(x).view(B, S, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # torch.nn.functional.scaled_dot_product_attention uses FA2 when available.
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.wo(out.transpose(1, 2).contiguous().view(B, S, -1))

block = MiniBlock()

for _ in range(50):
    seqlen = random.choice([256, 512, 1024, 2048, 4096])
    x = torch.randn(1, seqlen, H*D, dtype=torch.bfloat16, device=dev, requires_grad=True)
    torch.cuda.synchronize()
    t0 = time.time()
    y = block(x)
    y.sum().backward()
    torch.cuda.synchronize()
    print(f"seqlen {seqlen:5d}  step_s {time.time()-t0:.4f}")
```

Prints a clean monotonic relationship between seqlen and step time; 4096 is ~8–16× the cost of 256. In a 4-rank FSDP run, whichever rank gets the 4096 sequence dictates the step time.

**Fix (`scenario-05/fix.md`).**

```markdown
## Fix
Three options; pick one with trade-offs documented.

### Option A — Length-bucketed sampling (`LengthGroupedSampler`)
```python
sampler = LengthGroupedSampler(
    dataset["train"],
    batch_size=args.per_device_train_batch_size,
    lengths=dataset["train"]["length"],
)
```
Trade-off: adjacent batches are similar in length, so step time stabilizes.
Risk: if the dataset is sorted after bucketing, the LR schedule interacts with
batch content order; shuffle *within* each length group. Easiest fix; keeps
the batch size per rank at `per_device_train_batch_size`.

### Option B — Sequence packing (`SFTTrainer(packing=True)`)
Pack multiple short sequences into one fixed-length example with per-example
attention masks. Increases tok/s by 20–50% (no padding waste). Requires
end-of-example tokens and a packing-aware attention mask; TRL's `SFTTrainer`
handles this automatically when `packing=True`.

### Option C — Fixed-length pad-to-multiple
Pad every example to a fixed seqlen (e.g., 2048) and truncate longer ones.
Simplest; most wasteful (padding is wasted compute); only sensible when the
distribution is already tight around the chosen seqlen.

## Choice for this fix
Option B (`SFTTrainer(packing=True)`) with per-example end-of-text tokens.
Rationale: our dataset is ~60% short (<1024 tokens), so packing saves ~40% of
wasted tokens while stabilizing step time.

## Verification
Post-fix step-time distribution (same 200-step window):
    P50 5.1 s, P90 5.4 s, min 4.9 s, max 5.6 s. Variance 14% (was 100%+).
    `nvidia-smi dmon` shows steady 92–98% utilization across all ranks.
```

### 2.7 `summary.md` (worked)

```markdown
# Cross-scenario summary

| # | Scenario | Family | Diagnosis speed | Diagnostic tool | Fix category |
|---|---|---|---|---|---|
| 1 | OOM step 0 | Activation OOM | <5 min from log | OOM traceback + chapter 4 allocation math | Enable FA2 / enable grad checkpointing |
| 2 | OOM step 200 (drift) | Memory leak (Python-side) | 15–30 min | Per-step mem log + `torch.cuda.memory._record_memory_history` | Drop the retained reference; `.detach().cpu()` if the tensor is needed |
| 3 | NCCL timeout | Peer disappeared | <5 min from log | `NCCL_DEBUG=INFO` + which rank's log stopped | All-reduce the stop condition instead of per-rank early return |
| 4 | Silent hang | Code-path divergence on collective | 5–30 min | `py-spy dump` every rank at the same wall-clock moment | Make the collective unconditional; move one-sided work behind a barrier |
| 5 | Load imbalance | Variable-length sequences | 15–30 min | Per-step wall + per-batch max seqlen | Length-bucketed sampler, sequence packing, or fixed-length pad |

## Honest self-assessment
Easiest: scenario 1. The OOM traceback is explicit — allocation size and the
operation name are both named. Combined with chapter 4's memory-component
table, the diagnosis is a 2-minute task.

Hardest: scenario 2. The drift is slow; without a per-step memory log and
without knowing to reach for `memory._record_memory_history`, students bisect
step numbers and lose an hour. The fix is trivial once localized; the
localization is the whole exercise.

Most "I should have seen this immediately" moment: scenario 4. The signature
is unmistakable once you've seen it (processes resident, GPU util 0, no error),
but a first-time reader can lose ~30 minutes inspecting `nvidia-smi` output
before reaching for `py-spy`.

## Escalation notes
| # | Fix myself / escalate? | Evidence to attach on escalation |
|---|---|---|
| 1 | Fix myself. | n/a |
| 2 | Fix myself. If leak is in a library (e.g., TRL metric accumulator), escalate to that library with the `memory_viz` screenshot. |
| 3 | Fix myself (the intentional exit is on my side). If the silent exit is Xid 79 or spot-preemption, escalate to platform: attach the silent rank's last stdout line, `dmesg` grep for `NVRM`/`Xid`, and `nvidia-smi` snapshot. |
| 4 | Fix myself. |
| 5 | Fix myself. If `LengthGroupedSampler` is unavailable in the local TRL pin, request a dependency bump from the platform team; attach the pre/post step-time variance numbers as justification. |
```

### 2.8 Directory layout (as delivered)

```
outputs/exercise-04/
├── train.py                                 # baseline, unmodified across scenarios
├── requirements.txt                         # ex01 pins + py-spy
├── scripts/
│   └── launch_debug.sh
├── scenario-01/
│   ├── injection.diff
│   ├── log.txt
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
├── scenario-02/
│   ├── injection.diff
│   ├── log.txt
│   ├── memory_viz_screenshot.png
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
├── scenario-03/
│   ├── injection.diff
│   ├── log.txt
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
├── scenario-04/
│   ├── injection.diff
│   ├── log.txt
│   ├── nvidia-smi.txt
│   ├── pyspy_dump.txt
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
├── scenario-05/
│   ├── injection.diff
│   ├── log.txt
│   ├── nvidia-smi-dmon.log
│   ├── diagnosis.md
│   ├── reproducer.py
│   └── fix.md
└── summary.md
```

## 3. Validation steps

1. **Launch every scenario with the diagnostic env vars set.** `NCCL_DEBUG=INFO`, `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Scenario 3 adds `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180`. Without these, the logs are missing the detail needed to diagnose.
2. **Per-rank, prefixed, per-step logging.** `[rank X] step N loss L mem_gb M max_seq_len S` is the format. Scenarios 2 and 5 are not diagnosable without per-step memory and per-batch max-seqlen respectively.
3. **Reproducer is standalone and short.** Every `reproducer.py` imports only `torch`, `torch.distributed`, and stdlib. Any that depend on `accelerate` / `transformers` / `datasets` are not reproducers; they are re-runs of the training script. Target 20–50 lines; hard ceiling 100.
4. **Reproducer actually reproduces.** Run each in isolation and confirm it exhibits the same failure family (not necessarily the same stack, but the same signature: OOM, drift, timeout, hang, step-time variance).
5. **Diagnosis cites log evidence.** Each `diagnosis.md` quotes specific log lines or stack frames that prove the failure family. "It's OOM" is not acceptable; "the OOM traceback shows a 12.5 GB allocation during forward on the attention-score tensor" is.
6. **Scenario 2's memory_viz screenshot is present and interpretable.** The screenshot shows a growing allocation attributed to the callback frame, not just a blob of memory.
7. **Scenario 4's py-spy dumps cover every rank at the same wall-clock moment.** A shell-wrapped `pgrep -f ... | xargs py-spy dump` loop is the correct form; single-rank dumps miss the whole point.
8. **Scenario 5's nvidia-smi-dmon log covers at least 60 seconds of steady-state training.** The uneven-utilization signature is only visible across a window long enough to cover multiple step cycles.
9. **Fix is verified with evidence.** Each `fix.md` contains a post-fix log excerpt, memory number, or step-time distribution that proves the job now runs. "It works now" is not a verification.
10. **Summary table is complete.** Every cell filled; escalation notes name specific evidence to attach.

## 4. Rubric for grading

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **All 5 scenarios attempted** | Fewer than 5 attempted | All 5 attempted but one is missing the reproducer or the fix | All 5 attempted with injection.diff + log.txt + diagnosis.md + reproducer.py + fix.md |
| **Reproducers are minimal** | Reproducer is >200 lines or re-runs the training script | Reproducer is standalone but >100 lines or very slow | Each reproducer is <100 lines, `torch`+stdlib only, reproduces in under 30 seconds |
| **Diagnosis cites log evidence** | "It's OOM" with no specifics | Log evidence present for 3–4 scenarios, generic for the rest | Every scenario names the specific log line or stack frame that proves the family |
| **Scenario 4 py-spy discipline** | No py-spy or single-rank only | py-spy for every rank but not simultaneously | Simultaneous py-spy dump of every rank with the shell wrapper, correctly interpreted |
| **Summary + escalation** | Table missing or generic self-assessment | Table complete, self-assessment thin or escalation notes generic | Table + honest self-assessment + per-scenario escalation decision with specific evidence named |

**Pass threshold:** 7/10, with no single dimension at 0 and diagnosis at ≥ 1.

## 5. Common mistakes

- **Launching without the diagnostic env vars.** The default `NCCL_DEBUG=WARN` + 30-minute watchdog timeout make scenarios 3 and 4 nearly impossible to diagnose — students wait 30 minutes to see the error. Always set the four env vars.
- **Treating a re-run of the training script as a reproducer.** A reproducer is 20–50 lines, standalone, under 10 seconds. If yours is 500 lines, you did not reproduce the failure; you re-ran the training script.
- **Diagnosing scenario 3 as "OOM then hang" because `TORCH_NCCL_ASYNC_ERROR_HANDLING` is off.** With the flag off, the exiting rank's exit is masked by the eventual OOM-looking NCCL error on the survivors, and scenario 2 and scenario 3 look identical. With the flag on, the survivors cleanly abort with a collective-timeout message that names the missing rank.
- **Taking a py-spy dump of only rank 0 in scenario 4.** Code-path divergence is a *comparative* signature; you need stacks from every rank at the same wall-clock moment to see that some ranks are in `barrier` and others are in `forward`. Single-rank dumps tell you nothing.
- **Pattern-matching scenario 2 to a framework bug.** FSDP and ZeRO-3 do not leak memory on their own; if memory grows monotonically on only one rank, the leak is Python-side. Look for `.append(...)` / `_history` / `wandb.log(...)` patterns in the training loop and callbacks first.
- **Scenario 5 fix: "we fixed it with `torch.compile`."** `torch.compile` does not solve load imbalance; variable-length sequences still force ranks to wait on the slowest. Compile may even make it worse via silent recompiles. The correct fix is in the data pipeline (bucketing, packing, or fixed-length padding).
- **Not documenting the pre/post variance in scenario 5.** The fix's whole value proposition is "step-time variance drops from 100% to 15%." Without the numbers, the fix is unverifiable.
- **Reporting scenario 1 memory breakdown qualitatively.** The chapter 4 memory components are enumerable (weights, grads, optimizer, activations, comm buffers, framework overhead); the diagnosis should name which specific component blew up (activations) and *why* (FA2 off → explicit `[b, a, s, s]` attention-score tensor).
- **Fixing scenario 3 by extending the heartbeat timeout.** The timeout fired because a rank disappeared; a longer timeout just means the same failure arrives later. The fix is on the exiting rank, not on the watchdog.
- **Skipping the summary table.** It is the artifact that captures the muscle memory the exercise builds. A student who skips it has done the work and discarded the learning.

## 6. References

- PyTorch Distributed — NCCL process-group, debugging env vars, `TORCH_DISTRIBUTED_DEBUG`, `TORCH_NCCL_ASYNC_ERROR_HANDLING`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`. <https://pytorch.org/docs/stable/distributed.html>
- PyTorch CUDA memory diagnostics — `torch.cuda.memory._record_memory_history`, `_dump_snapshot`, and the memory viewer at <https://pytorch.org/memory_viz>. <https://pytorch.org/docs/stable/torch_cuda_memory.html>
- `py-spy` — sampling profiler for Python, used here for stack dumps on live processes. <https://github.com/benfred/py-spy>
- NVIDIA NCCL documentation — `NCCL_DEBUG=INFO` output and interpretation, Xid error codes. <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>, <https://docs.nvidia.com/deploy/xid-errors/index.html>
- Hugging Face Transformers / TRL — `LengthGroupedSampler`, `SFTTrainer(packing=True)`, data-pipeline options for load-imbalance fixes. <https://huggingface.co/docs/transformers/main_classes/trainer>, <https://huggingface.co/docs/trl/sft_trainer>
- Dao, "FlashAttention" / "FlashAttention-2" — why FA2 avoids materializing the `[b, h, s, s]` attention-score tensor (scenario 1). <https://arxiv.org/abs/2205.14135>, <https://arxiv.org/abs/2307.08691>
- NIST AI Risk Management Framework — *Manage 4.3 (incident response)* and *Measure 4.2 (operational monitoring)* frame the diagnostic reflex this exercise builds. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — *ML05 Model Theft* / *ML06 AI Supply Chain Vulnerabilities* frame the "silent rank disappearance" scenario against resource-exhaustion and runtime-integrity threats. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — tactics catalog; the diagnostic artifacts (logs, py-spy dumps, dmon traces) are the evidence objects an incident-response flow would collect. <https://atlas.mitre.org/>
