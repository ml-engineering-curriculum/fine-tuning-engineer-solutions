# mod-108 · exercise-01 — `lm-evaluation-harness` Baseline vs. Tuned — Solution

## 1. Solution overview

The deliverable is a **reproducible benchmark comparison** between a base model and its fine-tuned descendant, using EleutherAI's `lm-evaluation-harness` so the same task set, prompt templates, and scoring code run against both checkpoints. The output is not a single number — it is a per-task table with 95% confidence intervals, a sign-of-delta and significance verdict per task, and a short analysis defending whichever tasks moved in the "wrong" direction.

Four ideas carry the exercise:

- **Same harness, same tasks, same seeds.** Benchmark numbers are only comparable when the prompt template, few-shot sampling order, and scoring function are byte-identical. `lm-evaluation-harness` enforces this by versioning tasks (e.g. `mmlu` vs `mmlu_pro`) and seeding the few-shot sampler. If the base and tuned runs use different task versions or different `--num_fewshot`, the delta is noise.
- **Point estimates without CIs are not comparable.** A +1.2% MMLU improvement on a 14k-item benchmark is a real signal; the same +1.2% on a 500-item benchmark is inside the standard-error band. The harness emits `stderr` for every metric; the solution computes 95% CIs (approximately `± 1.96 · stderr`) and uses a two-proportion z-test before claiming a win.
- **Expect regressions — budget for them.** Fine-tuning on a narrow distribution almost always regresses broad-knowledge benchmarks (MMLU, ARC) by 1–5 points while improving task-specific ones. A report that does not name the regressed tasks and defend them has missed the point.
- **The tuned checkpoint's chat template matters.** If the base model is evaluated with no chat template and the tuned model is evaluated with one, the comparison is confounded by prompt format, not by weights. The solution runs both under the tuned model's chat template (or both without one) and documents the choice.

The worked example evaluates `meta-llama/Llama-3.2-3B` (base) against a fine-tuned variant `./runs/llama-3.2-3b-sft` across seven tasks: `mmlu`, `arc_challenge`, `hellaswag`, `winogrande`, `truthfulqa_mc2`, `gsm8k`, and a task-specific held-out slice registered as a custom task. The pattern generalizes to any (base, tuned) pair.

## 2. Worked implementation

### 2.1 Environment and pins

```text
# requirements.txt
lm-eval==0.4.5
torch==2.4.1
transformers==4.45.2
accelerate==1.0.1
vllm==0.6.3                # optional; used for the --model vllm path
datasets==3.0.1
sentencepiece==0.2.0
```

Re-check the `lm-evaluation-harness` release notes before pinning; task definitions and defaults change between minor versions, and a task-version drift between runs silently invalidates the comparison. The version used must be recorded in `report.md` (see §2.6).

### 2.2 Choose the task set up front and freeze it

```yaml
# tasks.yaml — frozen before any run. Any change invalidates prior numbers.
harness_version: "0.4.5"
tasks:
  - name: mmlu                       # knowledge breadth
    num_fewshot: 5
    limit: null
  - name: arc_challenge              # reasoning on grade-school science
    num_fewshot: 25
    limit: null
  - name: hellaswag                  # commonsense sentence completion
    num_fewshot: 10
    limit: null
  - name: winogrande                 # pronoun resolution
    num_fewshot: 5
    limit: null
  - name: truthfulqa_mc2             # truthfulness, multi-choice v2
    num_fewshot: 0
    limit: null
  - name: gsm8k                      # grade-school math, exact-match
    num_fewshot: 8
    limit: null
  - name: internal_task_v1           # custom task; see §2.5
    num_fewshot: 0
    limit: 1000                      # fixed subset to keep latency sane
seeds:
  fewshot_seed: 1234
  numpy_seed: 1234
  torch_seed: 1234
```

`fewshot_seed`, `numpy_seed`, and `torch_seed` are the three the harness uses to make the few-shot sample order and any sampling-based scoring deterministic. They must be identical across base and tuned runs.

### 2.3 Run the base model

```bash
# run_base.sh
set -euo pipefail
export HF_DATASETS_CACHE=./cache
mkdir -p results/base

lm_eval \
  --model hf \
  --model_args "pretrained=meta-llama/Llama-3.2-3B,dtype=bfloat16,attn_implementation=flash_attention_2,parallelize=False" \
  --tasks mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k,internal_task_v1 \
  --num_fewshot 5 \
  --batch_size auto:4 \
  --seed 1234 \
  --fewshot_random_seed 1234 \
  --output_path results/base \
  --log_samples                          # writes per-sample jsonl for CI/regression triage
```

`--num_fewshot 5` on the CLI is the *default* applied to any task that does not override it in `tasks.yaml`. The harness uses the task-native default when specified in the task YAML, so running `--tasks mmlu,...` with the standard tasks already yields the num-fewshot in the table above.

`--batch_size auto:4` starts from a target of 4 and binary-searches upward per task until OOM. Deterministic throughput, deterministic numbers.

`--log_samples` writes `samples_<task>_<timestamp>.jsonl` — one row per eval example with the prompt, continuation, log-likelihoods, and target. These files are what the regression triage in §3 consumes.

### 2.4 Run the tuned model, under its chat template

```bash
# run_tuned.sh
set -euo pipefail
export HF_DATASETS_CACHE=./cache
mkdir -p results/tuned

lm_eval \
  --model hf \
  --model_args "pretrained=./runs/llama-3.2-3b-sft,dtype=bfloat16,attn_implementation=flash_attention_2,parallelize=False" \
  --tasks mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2,gsm8k,internal_task_v1 \
  --num_fewshot 5 \
  --batch_size auto:4 \
  --seed 1234 \
  --fewshot_random_seed 1234 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path results/tuned \
  --log_samples
```

Two flags to defend explicitly in the report:

- `--apply_chat_template`: use the tokenizer's chat template so the tuned model receives the format it was trained on. On a base model this would degrade MMLU by several points; on a chat-tuned model, the opposite. If both base and tuned use the base-model prompt format (no chat template), the comparison measures weights only, which is also defensible — the choice must be *stated* and *consistent*.
- `--fewshot_as_multiturn`: renders few-shot examples as user/assistant turns instead of a flat text concatenation. This is the right default for chat models; without it, the few-shot context is dumped as one user turn and the model may ignore the demonstrations.

**Chat template rule of thumb**: if the tuned model's chat template is set, use `--apply_chat_template` on both runs (the base model's chat template is usually a no-op) or on neither, but *the same decision on both*. Document which, and why.

### 2.5 Register the custom held-out task

The internal task is a jsonl of `{"doc_id": ..., "question": ..., "choices": [...], "answer": <index>}` records. Register it as an lm-eval task YAML:

```yaml
# lm_eval_tasks/internal_task_v1.yaml
task: internal_task_v1
dataset_path: json
dataset_kwargs:
  data_files: ./data/internal_eval_v1.jsonl
test_split: train                        # the file is the test set; there's no train
output_type: multiple_choice
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_choice: "{{choices}}"
doc_to_target: "{{answer}}"              # integer index into choices
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
num_fewshot: 0
metadata:
  version: 1.0
```

Register the directory:

```bash
export LM_EVAL_TASK_CONFIG_PATH=$(pwd)/lm_eval_tasks
lm_eval --tasks internal_task_v1 ...
```

Version the task YAML. The `metadata.version` field moves whenever the dataset, prompt, or scoring changes, and a bumped version produces a new task ID in the harness — old numbers are not comparable. This versioning is the on-disk analog of the "freeze the eval set" rule.

### 2.6 Compare the two runs

```python
# compare.py — reads the two JSON outputs and emits the comparison table.
from __future__ import annotations
import json, math, pathlib

BASE = pathlib.Path("results/base")
TUNED = pathlib.Path("results/tuned")

# Each lm-eval run writes a results JSON named by model + timestamp.
def _latest_results(root: pathlib.Path) -> dict:
    # Find the newest results_*.json in the output tree.
    candidates = sorted(root.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no results_*.json under {root}")
    return json.loads(candidates[-1].read_text())

def _primary_metric(task: str, metrics: dict) -> tuple[str, float, float]:
    """Return (metric_name, value, stderr) using the harness's convention."""
    # For multiple_choice tasks the harness reports acc and acc_norm;
    # default to acc_norm for ARC/HellaSwag, acc elsewhere.
    prefer = {
        "arc_challenge": "acc_norm,none",
        "hellaswag":     "acc_norm,none",
    }.get(task, None)
    # Keys look like "acc,none" and "acc_stderr,none".
    if prefer and prefer in metrics:
        return prefer.rsplit(",", 1)[0], metrics[prefer], metrics[prefer.replace("acc_norm", "acc_norm_stderr")]
    for name in ("acc,none", "exact_match,strict-match", "exact_match,flexible-extract"):
        if name in metrics:
            stderr_key = name.replace(name.split(",")[0], f"{name.split(',')[0]}_stderr")
            return name.split(",")[0], metrics[name], metrics.get(stderr_key, float("nan"))
    raise KeyError(f"no recognised metric for task {task}: {list(metrics)}")

def _two_proportion_z(p1: float, se1: float, p2: float, se2: float) -> float:
    """z-score for H0: p1 == p2, pooled SE via independent-SEs approximation."""
    denom = math.sqrt(max(1e-12, se1 * se1 + se2 * se2))
    return (p2 - p1) / denom

def main() -> None:
    base = _latest_results(BASE)
    tuned = _latest_results(TUNED)
    rows = []
    for task in base["results"].keys() & tuned["results"].keys():
        bname, bval, bse = _primary_metric(task, base["results"][task])
        tname, tval, tse = _primary_metric(task, tuned["results"][task])
        assert bname == tname, (task, bname, tname)
        delta = tval - bval
        z = _two_proportion_z(bval, bse, tval, tse)
        # Two-sided p via normal approximation.
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        rows.append({
            "task": task,
            "metric": bname,
            "base": round(bval, 4),
            "tuned": round(tval, 4),
            "delta_abs": round(delta, 4),
            "delta_rel_pct": round(100 * delta / max(bval, 1e-9), 2),
            "base_ci95": [round(bval - 1.96 * bse, 4), round(bval + 1.96 * bse, 4)],
            "tuned_ci95": [round(tval - 1.96 * tse, 4), round(tval + 1.96 * tse, 4)],
            "z": round(z, 2),
            "p": round(p, 4),
            "verdict": "win" if z > 1.96 else ("loss" if z < -1.96 else "flat"),
        })
    rows.sort(key=lambda r: r["task"])
    out = {
        "harness_version": base.get("config", {}).get("harness_version", "unknown"),
        "base_model": base.get("config", {}).get("model_source", "base"),
        "tuned_model": tuned.get("config", {}).get("model_source", "tuned"),
        "rows": rows,
    }
    pathlib.Path("results/comparison.json").write_text(json.dumps(out, indent=2))
    _render_markdown(rows)

def _render_markdown(rows: list[dict]) -> None:
    lines = ["| Task | Metric | Base | Tuned | Δ abs | Δ %  |  z  |  p   | Verdict |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| `{r['task']}` | {r['metric']} | {r['base']:.3f} | {r['tuned']:.3f} | "
            f"{r['delta_abs']:+.3f} | {r['delta_rel_pct']:+.1f}% | {r['z']:+.2f} | {r['p']:.3f} | {r['verdict']} |")
    pathlib.Path("results/comparison.md").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
```

### 2.7 Worked results table (illustrative; your numbers will differ)

Reference numbers from running the two scripts above on 1× A100-80GB against Llama-3.2-3B base and a hypothetical 100k-example SFT on a customer-support task set. Treat the digits as illustrative; the examinable claims are the direction of each delta and the z-score threshold.

| Task | Metric | Base | Tuned | Δ abs | Δ %  |  z  |  p   | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `mmlu`           | acc      | 0.541 | 0.526 | −0.015 | −2.8% | −2.34 | 0.019 | loss |
| `arc_challenge`  | acc_norm | 0.434 | 0.431 | −0.003 | −0.7% | −0.21 | 0.834 | flat |
| `hellaswag`      | acc_norm | 0.751 | 0.743 | −0.008 | −1.1% | −1.31 | 0.190 | flat |
| `winogrande`     | acc      | 0.682 | 0.681 | −0.001 | −0.1% | −0.09 | 0.928 | flat |
| `truthfulqa_mc2` | acc      | 0.397 | 0.429 | +0.032 | +8.1% | +2.19 | 0.029 | win |
| `gsm8k`          | exact_match | 0.267 | 0.258 | −0.009 | −3.4% | −0.58 | 0.562 | flat |
| `internal_task_v1` | acc    | 0.512 | 0.731 | +0.219 | +42.8% | +9.86 | 0.000 | win |

**Reading the table**:

- The intended-task delta (`internal_task_v1`) is large and significant — the fine-tune works on what it was trained for.
- MMLU regressed significantly (|z| > 1.96, p < 0.05). This is expected on narrow SFT; it must be *named* in the report, not swept under "general improvements."
- ARC/HellaSwag/WinoGrande/GSM8K deltas are not significant. "No measurable change" is the correct finding; a report that claims a win on `gsm8k` from −0.009 absolute is reading noise.
- `truthfulqa_mc2` improved, which is not an assumed outcome of SFT and deserves a sentence of explanation (safety-adjacent training data? Chat template effect? Investigate before claiming credit.)

### 2.8 Directory layout (as delivered)

```
outputs/exercise-01/
├── tasks.yaml
├── lm_eval_tasks/
│   └── internal_task_v1.yaml
├── run_base.sh
├── run_tuned.sh
├── compare.py
├── results/
│   ├── base/
│   │   ├── results_<ts>.json
│   │   └── samples_<task>_<ts>.jsonl   # one per task
│   ├── tuned/
│   │   ├── results_<ts>.json
│   │   └── samples_<task>_<ts>.jsonl
│   ├── comparison.json
│   └── comparison.md
├── report.md
└── env.txt                              # pip freeze and GPU info
```

### 2.9 `report.md` (skeleton the student fills in with measured numbers)

```markdown
# Baseline vs. Tuned — report

## Setup
- Harness: lm-evaluation-harness v0.4.5 (commit abc123)
- Base model: meta-llama/Llama-3.2-3B
- Tuned model: ./runs/llama-3.2-3b-sft (SFT, 100k rows, 3 epochs)
- Chat template: applied to tuned, applied to base (base template is no-op) — same decision on both runs
- Seeds: fewshot=1234, numpy=1234, torch=1234
- Hardware: 1× A100-80GB; bf16; FA2 on
- Date: 2026-11-17

## Results
(see comparison table in §2.7)

## Wins
- `internal_task_v1`: +21.9 pts absolute, z=+9.86. The fine-tune achieved its primary goal.
- `truthfulqa_mc2`: +3.2 pts absolute, z=+2.19. Needs investigation; hypothesis: the SFT corpus includes refusal / safety-adjacent demonstrations that transfer to TruthfulQA-style distractors.

## Regressions (must be defended)
- `mmlu`: −1.5 pts absolute, z=−2.34, p=0.019. Expected direction given the narrow SFT corpus; magnitude is within the published literature's "1–3 pts" band for narrow SFT on a 3B base. Mitigation: mixing in a slice of general-instruction data in the next run (0.5–1.0× the task data); re-measure MMLU.

## Flat (not-significant) deltas
- `arc_challenge`, `hellaswag`, `winogrande`, `gsm8k`: |z| < 1.96. No claim of change in either direction.

## Open questions / next steps
1. Why did truthfulqa_mc2 improve? Inspect `samples_truthfulqa_mc2_<ts>.jsonl` from both runs; compare per-item deltas.
2. Can the MMLU regression be closed by adding 10% general-instruction data?
3. Confirm the `internal_task_v1` lift reproduces on a held-out slice the training data has never seen (exercise 02).
```

## 3. Validation steps

1. **Harness and task versions match across runs.** `results/base/results_*.json` and `results/tuned/results_*.json` both report the same `versions` block (e.g. `"mmlu": 2.0`). A task-version mismatch is a hard fail; rerun under one consistent version.
2. **Seeds match.** `--seed`, `--fewshot_random_seed`, and the task-level seed are identical in `run_base.sh` and `run_tuned.sh`. If different, the few-shot sampler produces different shots per example and the delta is unmeasurable.
3. **Chat-template decision is consistent and documented.** If `--apply_chat_template` is set on tuned, it is set on base, and the report names the choice. Mismatched template flags are the #1 source of spurious wins.
4. **Every task has a stderr in the JSON.** Grep `_stderr` in both results JSONs for every reported task. Missing stderr means the comparison's z-score cannot be computed and the verdict is "unknown," not "win."
5. **`--log_samples` is on.** `samples_<task>_<ts>.jsonl` files exist for each task in both runs. They are what per-item regression triage (exercise 05) consumes.
6. **Point estimate ±1.96·stderr brackets the base from tuned on every row claimed as a win.** If the tuned CI95 overlaps the base point estimate, the win is noise. The `verdict` field in `comparison.md` is derived from the z-test, not from "the number got bigger."
7. **Regression rows are explicitly addressed.** Any task with `verdict == loss` has a one-paragraph explanation in `report.md` naming the hypothesised cause and a planned mitigation. Silent regressions are the exercise's primary anti-pattern.
8. **Custom task YAML is versioned.** `metadata.version` in `lm_eval_tasks/internal_task_v1.yaml` matches the version used in both runs. Any change to prompt, data, or scoring bumps the version and re-runs are needed on both base and tuned.
9. **Environment is captured.** `env.txt` contains `pip freeze` output and `nvidia-smi` headers; the harness commit hash is noted in `report.md`.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Both runs are reproducible from the scripts** | A reviewer cannot rerun either because flags, seeds, or env are missing | Scripts exist and run, but one or more flags differ between base and tuned | `run_base.sh` and `run_tuned.sh` are identical except for `--model_args` and `--output_path`; task set, seeds, batching, and chat-template decision match across both |
| **Task coverage** | Fewer than 3 standard benchmarks; no task-specific held-out task | At least 3 standard benchmarks and the internal task, but no attention to known-regression candidates (MMLU/ARC) | Full standard set (knowledge, reasoning, commonsense, truthfulness, math) plus the internal task; task choice is justified in `report.md` |
| **Statistical rigor** | Deltas reported without CIs or stderr | CIs reported, but win/loss verdicts are called from raw point estimates | Every row has CI95 and z-score; verdict column uses the z-test threshold; `report.md` explicitly lists "flat" rows as "no measurable change" |
| **Regression analysis** | Regressions not mentioned | Regressions listed with no cause or mitigation | Each regression named, hypothesis for cause given, concrete mitigation proposed and tied to the next-iteration plan |
| **Documentation + versioning** | Harness version, seeds, or task YAML version missing | Some of the above; chat-template decision undefended | All of harness version/commit, task-YAML version, seeds, chat-template decision, and env are captured; the comparison can be rerun 6 months later and reproduce bit-identically |

**Pass threshold:** 7/10, with no single dimension at 0 and the statistical-rigor dimension at 2.

## 5. Common mistakes

- **Running base without the chat template and tuned with it.** This is the most frequent source of inflated deltas. Pick one and apply it to both; defend the choice in the report.
- **Changing `--num_fewshot` between runs** (e.g. the base at 5-shot and the tuned at 0-shot "because the chat model doesn't need few-shot"). Few-shot count is part of the task definition; changing it between runs means two different benchmarks.
- **Comparing `acc` from the base to `acc_norm` from the tuned.** On multiple-choice tasks the harness reports both; `acc_norm` divides log-likelihood by continuation byte length and is the convention for ARC / HellaSwag. Pick the same metric per task across both runs.
- **Reading a delta inside the CI as a win.** +0.9% on MMLU at a 0.6% stderr is a +1.5 z-score — not significant. The two-proportion z-test exists for this exact case; use it.
- **Not freezing `tasks.yaml`.** A student who edits the task list between the base and tuned runs ships a comparison that is technically different benchmarks. Treat the YAML as immutable once run 1 starts.
- **Skipping `--log_samples`.** Without per-sample jsonl, regression triage becomes guessing. These files are cheap (a few MB per task) and are the only way to pivot from "MMLU regressed by 1.5 pts" to "MMLU regressed on the 'professional_medicine' subcategory specifically."
- **Treating MMLU, HellaSwag, and ARC as "the generalization check" uncritically.** Several standard benchmarks have documented contamination issues with pretraining corpora (see exercise 04). The reported delta on these benchmarks is a *floor* on regression; the actual generalization is measured on the held-out task.
- **Skipping the internal task.** Standard benchmarks are necessary but not sufficient; the internal task is the only one guaranteed to reflect what the fine-tune was *for*. A report that reports MMLU and nothing task-specific has not evaluated the tune.
- **Running one of the two models on a different `batch_size` or `dtype`.** bf16 vs fp16 vs fp32 produces slightly different log-likelihoods on long generations; keep both runs on the same `dtype`. Different `batch_size` can shift numbers at the fourth decimal via batched-padding effects; `--batch_size auto:4` on both avoids this.
- **Reporting numbers without harness commit hash and task versions.** Six months later, when someone asks "did we really improve on MMLU for the v2 model?", the comparison must be bit-reproducible. The harness version field in `report.md` is what makes that possible.

## 6. References

- EleutherAI `lm-evaluation-harness` — repository, usage docs, and the `--apply_chat_template` / `--fewshot_as_multiturn` / `--log_samples` flags referenced above. <https://github.com/EleutherAI/lm-evaluation-harness>
- Hugging Face Transformers — tokenizer chat-template documentation (`tokenizer.apply_chat_template`). <https://huggingface.co/docs/transformers/chat_templating>
- NIST AI Risk Management Framework — *Measure 2.3* (model performance measured against operational conditions) and *Measure 4.2* (operational monitoring) apply to the two-proportion-z verdicts and the regression table that gates release. <https://www.nist.gov/itl/ai-risk-management-framework>
- OWASP Machine Learning Security Top 10 — *ML06 AI Supply Chain Vulnerabilities* is the framing for pinning `lm-eval`, task YAML versions, and model hashes so results are reproducible by third-party auditors. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — technique catalog; the per-sample logging in `samples_*.jsonl` supports the detection and triage story around *Evade ML Model* / *Verify Attack* categories when the same harness is pointed at adversarial inputs. <https://atlas.mitre.org/>
- Hendrycks et al. — *Measuring Massive Multitask Language Understanding (MMLU)*. The canonical knowledge benchmark; used here only through the harness-registered task. <https://arxiv.org/abs/2009.03300>
- Clark et al. — *Think you have Solved Question Answering? Try ARC* (ARC-Challenge). <https://arxiv.org/abs/1803.05457>
- Zellers et al. — *HellaSwag: Can a Machine Really Finish Your Sentence?* <https://arxiv.org/abs/1905.07830>
- Sakaguchi et al. — *WinoGrande*. <https://arxiv.org/abs/1907.10641>
- Lin et al. — *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. <https://arxiv.org/abs/2109.07958>
- Cobbe et al. — *Training Verifiers to Solve Math Word Problems* (GSM8K). <https://arxiv.org/abs/2110.14168>
