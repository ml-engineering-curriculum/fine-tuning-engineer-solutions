# mod-103 · exercise-05 — Reproducibility and Training Report — Solution

## 1. Solution overview

A fine-tuned model that cannot be reproduced is an artifact, not a result. The exercise ends the module by turning the exercise-01 run into something a stranger — a new teammate, a legal reviewer, an auditor — can rebuild from committed files. The deliverable is two artifacts: (a) a locked-down training recipe that rebuilds the model to within a defined tolerance; (b) a `training_report.md` and a model card a reviewer can read in under five minutes.

Three ideas carry the exercise:

- **Reproducibility is a tolerance, not a boolean.** Full bitwise determinism is possible on a single GPU with specific flags set; cross-GPU or multi-process bitwise reproducibility is in practice unachievable. Declare the tolerance up front ("same final eval loss within 1e-3 on the same hardware; within 5e-3 across H100 SKUs") and measure against it.
- **A training report is the audit surface, not a lab notebook.** The report names inputs by hash, compute by SKU and dtype, outputs by checksum, and limitations by name. If a reader cannot tell whether the model is safe to deploy from the report alone, the report is incomplete.
- **Model cards are not optional.** They are the format NIST AI RMF, the EU AI Act, and the community conventions (Mitchell et al., 2019) all converge on for communicating intended use, training data, and limitations [MODELCARD][NIST-AI-RMF][EUAIACT].

The exercise re-runs the exercise-01 configuration three times under a reproducibility harness and produces one paper trail covering all three runs.

## 2. Worked implementation

### 2.1 The reproducibility contract (write this first)

Commit `REPRO.md` before touching training code:

```markdown
# Reproducibility contract — sft-1b-v1

## Tolerance tiers

1. **Same box, same seed (bitwise target).** On the same GPU SKU, the same CUDA / driver / `torch` version, the same `transformers` + `trl` versions, with deterministic flags on, two runs must produce:
   - identical `final/model.safetensors` SHA-256
   - per-step train_loss within |Δ| < 1e-6
   - identical eval_loss at every eval step
2. **Same box, different seed (statistical target).** Final eval_loss standard deviation across 3 seeds ≤ 5e-3. SFT-gain metric (exercise-04) within 0.5 absolute points.
3. **Different box (operational target).** Final eval_loss within 1e-2; qualitative panel (exercise-04) produces the same answers for 4 of 5 prompts.

## In-scope

- Tokenizer version pin, dataset snapshot hash, chat template string.
- Random seeds: `torch`, `numpy`, `random`, `transformers.set_seed`, `data_seed`.
- Numeric flags: `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, matmul precision, `tf32`.
- Compute surface: GPU SKU + driver + CUDA + `torch` + `flash-attn` versions.

## Out of scope

- Multi-node bitwise reproducibility (gather/reduce order depends on NCCL build).
- Cross-vendor reproducibility (ROCm vs. CUDA).
- Reproducibility across major `transformers` versions (chat template semantics change).
```

### 2.2 `env.lock` — the exact environment

A committed `requirements.txt` is not enough; `pip freeze` is not enough either because it loses CUDA metadata. Capture three files:

```bash
# capture_env.sh
python -V > env.python
pip freeze > env.pip.txt
conda list --explicit > env.conda.txt 2>/dev/null || true
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv > env.gpu.csv
python - <<'PY' > env.torch.json
import json, torch
print(json.dumps({
  "torch": torch.__version__,
  "cuda": torch.version.cuda,
  "cudnn": torch.backends.cudnn.version(),
  "git_versions": {
    "flash_attn": __import__("flash_attn").__version__ if "flash_attn" in dir(__import__("sys").modules) else None,
  },
}, indent=2))
PY
```

Pin `transformers`, `trl`, `datasets`, `tokenizers`, `peft`, `accelerate`, `torch`, `flash-attn`, `bitsandbytes` by *version-and-build*, not by `>=`. Any of these can change chat-template rendering or kernel selection between patch releases.

### 2.3 Deterministic training knobs

Add to the top of `train.py`:

```python
import os, random, numpy as np, torch
from transformers import set_seed

SEED = 42

# Python + numpy + torch RNGs.
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# Deterministic cuBLAS workspace: pick one, commit the value.
# This is required for `use_deterministic_algorithms(True)` to permit matmul on CUDA.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Deterministic kernels. Has a measurable perf cost (~5–15% on SFT workloads).
# Flip True for the audit run; leave False for iteration runs.
torch.use_deterministic_algorithms(True, warn_only=True)

# TF32 narrows precision on Ampere+; same ops on different SKUs may diverge at the
# ~1e-4 level. Turn it OFF for the bitwise-target audit run.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# PyTorch 2.x matmul precision: "highest" = fp32 matmul; "high" = TF32; "medium" = BF16-reduce.
torch.set_float32_matmul_precision("highest")
```

Commit these values. "Randomly chose a seed" and "ran with defaults" together are indistinguishable from "did not try to be reproducible."

### 2.4 Data snapshot — hash before the run

The dataset at `HuggingFaceH4/no_robots` is updated upstream; a run six months from now will not see the same rows unless the snapshot is pinned:

```python
# snapshot.py
import hashlib, json, pathlib
from datasets import load_dataset

def snapshot(dataset_id: str, revision: str, out: pathlib.Path):
    ds = load_dataset(dataset_id, revision=revision)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": dataset_id, "revision": revision, "splits": {}}
    for split, data in ds.items():
        local = out / f"{split}.jsonl"
        with local.open("w") as f:
            for ex in data:
                f.write(json.dumps(ex, sort_keys=True) + "\n")
        h = hashlib.sha256(local.read_bytes()).hexdigest()
        manifest["splits"][split] = {"path": str(local), "rows": len(data), "sha256": h}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    snapshot("HuggingFaceH4/no_robots",
             revision="main",            # pin to a commit SHA for the audit run
             out=pathlib.Path("data/no_robots_v1"))
```

Train against `data/no_robots_v1/train_sft.jsonl` by passing it to `datasets.load_dataset("json", data_files=...)`. Commit the `manifest.json` (not the data itself unless license permits).

### 2.5 Three-seed sanity

Run the same recipe at seeds `{42, 7, 2026}`. Expected shape:

| Seed | Steps | Train loss (final) | Eval loss (final) | SFT-gain metric |
|---|---|---|---|---|
| 42 | 594 | 1.12 | 1.10 | 3.84 |
| 7 | 594 | 1.13 | 1.11 | 3.81 |
| 2026 | 594 | 1.12 | 1.11 | 3.86 |

Report mean ± std explicitly. If `std(eval_loss) > 5e-3`, either the tolerance in the contract was wrong or an undeclared source of non-determinism is in play (likely: `DataLoader` worker shuffle, `flash-attn` kernel version, cuDNN benchmark autotuner). Investigate before publishing.

### 2.6 The training report

Target shape:

```markdown
# SFT Training Report — Llama-3.2-1B on no_robots (v1)

## Summary
One-paragraph summary of the run, its purpose, and the one number that matters for the intended use.

## Run manifest
- Run id, start/end times, git SHA (clean/dirty)
- Base model: `meta-llama/Llama-3.2-1B` @ commit `<sha>`
- Dataset: `HuggingFaceH4/no_robots` @ commit `<sha>`, snapshot `manifest.json` sha256
- Env: see `env.pip.txt`, `env.torch.json`, `env.gpu.csv`

## Compute
- GPU: 1x NVIDIA RTX 4090, 24 GB, driver 550.78, CUDA 12.4
- Precision: bf16 (`attn_implementation="sdpa"`)
- Deterministic flags: `use_deterministic_algorithms=True`, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Effective batch: 32. Tokens/step: ~65,536. Wall clock: 45m 11s.

## Hyperparameters
- LR 2e-5, cosine, warmup_ratio 0.03
- Weight decay 0.1, max_grad_norm 1.0
- Epochs 2, max_seq_length 2048
- Packing off, assistant_only_loss on
- Seed 42 (also ran seeds 7 and 2026, see §Reproducibility)

## Loss curve
Train 1.78 → 1.12; eval 1.69 → 1.10. (Plots: `loss.png`, `eval_loss.png`, `grad_norm.png`.)

## Reproducibility
Across seeds {42, 7, 2026}: eval_loss 1.107 ± 0.0025. Within tolerance tier 2 of REPRO.md.
Same-seed re-run on the same box produced identical `model.safetensors` sha256. Tier 1 met.

## Evaluation (from mod-103 exercise-04)
MMLU 0-shot 27.8 (base 28.5 → Δ -0.7); HellaSwag 48.9 → 48.4; GSM8K 7.5 → 7.1;
TruthfulQA MC1 25.9 → 25.4. See `eval/summary.json`.

## Known limitations
1. No safety alignment pass (mod-106 topic); model is a bare SFT.
2. English-only SFT corpus; non-English generation regresses relative to base.
3. 1B model; capability floor is low — do not interpret as a general assistant.
4. No contamination audit of SFT corpus against public benchmarks; see mod-102 exercise-02.
5. Full-parameter SFT; LoRA and QLoRA comparisons in mod-104.

## Reproduce
```
git checkout <sha>
bash capture_env.sh                           # diff against env.*.txt
python snapshot.py                             # or: datasets snapshot_download
SEED=42 python train.py
python smoke_generate.py runs/sft-1b-v1/final
lm_eval --model hf --model_args pretrained=runs/sft-1b-v1/final,dtype=bfloat16 \
   --tasks mmlu,hellaswag,gsm8k,truthfulqa_mc1 --output_path eval/final
```
```

### 2.7 The model card

The model card sits in the released model directory (`runs/sft-1b-v1/final/README.md`) and is written for downstream users who will never read the training report. Base it on the Mitchell et al. (2019) Model Cards spec and NIST AI RMF's "transparency" obligations [MODELCARD][NIST-AI-RMF]. A minimal model card:

```markdown
---
library_name: transformers
base_model: meta-llama/Llama-3.2-1B
license: llama3.2
language: en
pipeline_tag: text-generation
tags:
- sft
- instruction-tuning
- demonstration
---

# sft-1b-v1

**Base model:** `meta-llama/Llama-3.2-1B` (see upstream license).
**Training data:** `HuggingFaceH4/no_robots` (CC-BY-NC-4.0 — see [NOROBOTS]).
**Developed for:** curriculum demonstration (not production use).

## Intended use

- Education and research on SFT recipes at the 1B scale.
- Local chat assistants that tolerate low factual reliability.

## Out-of-scope / do-not-use

- Any safety-critical use. This model has no alignment pass; it will comply with
  prompts a safety-aligned model would refuse.
- Non-English tasks: the SFT corpus is English-only; Llama-3.2-1B's multilingual
  capacity has regressed relative to the base.
- Automated decision-making about people (hiring, lending, access).

## Training procedure

See `training_report.md` committed with this model. Short form:
Llama-3.2-1B base, bf16, LR 2e-5 cosine with 3% warmup, 2 epochs on
no_robots `train_sft`, assistant-only loss, no packing.

## Evaluation

| Benchmark | Base | This model | Δ |
|---|---|---|---|
| MMLU 0-shot | 28.5 | 27.8 | -0.7 |
| HellaSwag | 48.9 | 48.4 | -0.5 |
| GSM8K | 7.5 | 7.1 | -0.4 |
| TruthfulQA MC1 | 25.9 | 25.4 | -0.5 |

## Risks

- Hallucination: the model will invent plausible-sounding facts.
- Bias: inherits the base model's and the SFT corpus's biases.
- No red-team: no adversarial-prompt testing beyond the benchmarks above.

## License

Llama 3.2 Community License (base model). no_robots CC-BY-NC-4.0 (training data).
This release follows the more restrictive CC-BY-NC-4.0 for the SFT artifact.

## Citation

See `training_report.md` for the exact commit, data snapshot hash, and env lock.
```

### 2.8 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Tolerance tiers, not "it reproduces" | Bitwise identity is a strong claim; tiers make scope explicit | "Reproducible" boolean | Boolean claims fail silently as soon as any knob changes |
| Deterministic flags on for the audit run only | Determinism costs ~5–15% throughput | Always on | Costs training iteration speed; usually unnecessary during dev |
| Dataset snapshot + manifest.json | Upstream dataset drift is a common silent regression source | Pin by HF revision only | Revisions can be force-pushed; a committed hash is tamper-evident |
| Three seeds, not one or ten | One hides variance; ten rarely changes the conclusion vs three | Single-seed run | Hides the real uncertainty band |
| Separate training report and model card | Audit vs. downstream user have different needs | One combined document | Combined forces engineers to publish private details, or forces users to read engineering noise |
| Reference NIST AI RMF / Model Cards | Matches the widely-cited transparency conventions | Ad-hoc fields | Ad-hoc fields look sloppy and omit obligations like "intended use" and "out-of-scope" |
| Commit env.pip.txt + env.torch.json | `pip freeze` loses CUDA metadata; torch block captures it | `requirements.txt` only | Lax requirements file reproduces only by accident |

## 3. Validation steps

1. **Fresh clone rebuilds.** On a different box (or the same box with `rm -rf runs/` and a fresh virtualenv), running `pip install -r env.pip.txt && SEED=42 python train.py` must land at tier-3 tolerance of the original eval_loss. Any larger gap means a captured knob is missing.
2. **Snapshot hash integrity.** `sha256sum data/no_robots_v1/train_sft.jsonl` must equal the value in `manifest.json`. If it diverges, re-run the snapshot script — the local file was edited or appended.
3. **Model safetensors hash.** `sha256sum runs/sft-1b-v1/final/model.safetensors` from two bitwise-target runs is byte-identical. Different → a non-deterministic knob slipped in; find it by binary search of the deterministic flags.
4. **Report self-test.** The reproduce block in `training_report.md` is a shell script. Execute it on a clean machine; it must complete end-to-end. A reproduce block that cannot be shelled is aspirational, not documentation.
5. **Model card completeness.** Every field required by the Mitchell et al. spec is present and non-empty: Model Details, Intended Use, Out-of-scope, Training Data, Evaluation, Ethical Considerations [MODELCARD]. NIST AI RMF categories Govern/Map/Measure/Manage each have at least one answer in the combined report+card [NIST-AI-RMF].
6. **License trace.** Base model license, SFT dataset license, and released artifact license each cited with a URL. The final license is at least as restrictive as the most restrictive input.
7. **Reviewer dry run.** Hand the report + card (without the code) to a reader who did not run training. They can answer, within 5 minutes: (a) what the model is intended for, (b) what it is *not* intended for, (c) what the measured regression vs. base is, (d) how to rebuild it. If they cannot, revise.

## 4. Rubric for grading the learner's report

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Tolerance contract** | Not written, or "reproducible" as a boolean | Tiers named but no numeric threshold on any | ≥3 tiers with numeric thresholds and scope; thresholds match later measurements |
| **Determinism knobs** | None; relies on framework defaults | Some flags set, others defaulted | Seeds, `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, TF32, matmul precision, cuDNN — all set and committed |
| **Env + data pin** | `requirements.txt` only, no data hash | One of env/data pinned, not both | Full env capture (`pip`, `torch`, `gpu`) + dataset manifest with sha256 |
| **Training report** | Missing or <3 of the standard sections | Report present but reproduce block missing or non-executable | All sections, reproduce block executes end-to-end, limitations named |
| **Model card** | Not provided | Present but missing intended-use or out-of-scope | Full card matching the Mitchell-et-al. spec; license trace; evaluation table; risks called out |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Treating `set_seed(42)` as reproducibility. It covers the `random`/`numpy`/`torch` CPU RNGs and nothing about kernel non-determinism on GPU.
- Omitting `CUBLAS_WORKSPACE_CONFIG`. With `use_deterministic_algorithms(True)` and CUDA, this env var is required or PyTorch raises a runtime error on matmul.
- Leaving TF32 on for the audit run. TF32 is non-deterministic across SKUs at the ~1e-4 level — tier-1 bitwise reproducibility is unattainable with it on.
- `pip freeze` as "the environment." Loses CUDA/cuDNN/flash-attn build info; a fresh install from a frozen file can still produce different kernel code.
- Pinning a HuggingFace dataset by `revision="main"`. `main` is a moving reference; snapshot to a commit SHA or to a local file whose hash is committed.
- Writing the report after the run instead of a template committed before. The post-hoc report inevitably reflects the run's biases.
- Combining training report and model card. The audit reader and the downstream user have different needs; one document optimizes for neither.
- Omitting license trace. If base model is Llama 3.2 Community License and training data is CC-BY-NC-4.0, the released artifact cannot be redistributed under Apache-2.0.
- Claiming "same loss, so reproducible" on a single-seed re-run. Loss is a scalar; seeds move it by a few milli-nats even on bitwise-identical runs due to floating-point reduction order.
- Measuring reproducibility only on train loss. The downstream artifact is the weights; `sha256sum model.safetensors` is the honest tier-1 check.

## 6. References

- [MODELCARD] Mitchell et al., "Model Cards for Model Reporting," ACM FAT* 2019. <https://arxiv.org/abs/1810.03993>
- [DATASHEETS] Gebru et al., "Datasheets for Datasets," CACM 2021 — companion standard for dataset documentation. <https://arxiv.org/abs/1803.09010>
- [NIST-AI-RMF] NIST AI Risk Management Framework (AI RMF 1.0) — Govern 1.5 (documentation), Measure 2 (performance metrics), Manage 3 (ongoing monitoring). <https://www.nist.gov/itl/ai-risk-management-framework>
- [EUAIACT] European Parliament, "Artificial Intelligence Act," 2024 — Articles on transparency and technical documentation obligations for general-purpose AI models. <https://eur-lex.europa.eu/eli/reg/2024/1689/oj>
- [L3] Meta, "Llama 3.2" model card + Community License. <https://huggingface.co/meta-llama/Llama-3.2-1B>
- [NOROBOTS] HuggingFace H4, "No Robots" dataset (CC-BY-NC-4.0). <https://huggingface.co/datasets/HuggingFaceH4/no_robots>
- [TRL] HuggingFace TRL — `SFTTrainer` / `SFTConfig`. <https://huggingface.co/docs/trl/main/en/sft_trainer>
- [HFTRAIN] HuggingFace Transformers — `Trainer` reproducibility notes (`set_seed`, `data_seed`, `full_determinism`). <https://huggingface.co/docs/transformers/main/en/trainer#reproducibility>
- [TORCH-REPRO] PyTorch, "Reproducibility" documentation (`use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`). <https://pytorch.org/docs/stable/notes/randomness.html>
- [BF16] Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training," 2019. <https://arxiv.org/abs/1905.12322>
- [TF32] NVIDIA, "TensorFloat-32 in the NVIDIA Ampere Architecture," 2020. <https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/>
- OWASP Machine Learning Security Top 10 — ML-06 AI Supply Chain Vulnerabilities (environment lockdown is a concrete mitigation). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — "ML Supply Chain Compromise" and "ML Model Provenance" techniques. <https://atlas.mitre.org/>
