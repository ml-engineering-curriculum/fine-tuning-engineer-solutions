# mod-103 · exercise-04 — Catastrophic Forgetting Investigation — Solution

## 1. Solution overview

A fine-tuned model that is better on your target task and worse on general knowledge is not strictly a better model. The exercise forces the learner to measure the *regression* — what the base model used to be able to do that the fine-tuned model can no longer do — and to run two mitigations against it. The goal is a comparison matrix a reviewer can read in one screen.

Three invariants carry the exercise:

- **Baselines are frozen before any training.** Evaluate the base model first, commit the numbers, then fine-tune. Evaluating the base after SFT — "we ran a quick check" — leaks dependencies (same tokenizer state, same cache, same seed drift) and makes the comparison invalid.
- **Regressions are measured on *held-out* general capabilities, not on the SFT corpus itself.** If the eval set overlaps the SFT training set, forgetting is hidden by memorization. Reuse the MMLU / HellaSwag / GSM8K / TruthfulQA subset logic from mod-108 — the point here is to use benchmarks the SFT data never touched [MMLU] [HELLA] [GSM8K] [TRUTHFULQA].
- **Mitigations are framed as trade-offs, not fixes.** Lower LR, replay with pretraining-style data, and LoRA each reduce forgetting by giving up some SFT gain. The deliverable names the exchange rate — "mitigation M recovered X% of forgetting at the cost of Y% of SFT gain" — not a winner.

The worked investigation reuses the `Llama-3.2-1B` + `no_robots` setup from exercise-01 as the baseline SFT run, then runs two mitigations: (M1) lower LR + fewer epochs, (M2) replay with `allenai/tulu-3-sft-mixture` (a general-capabilities instruction mix) interleaved at 30% [TULU3].

## 2. Worked implementation

### 2.1 The comparison matrix

Build this before any training. Every cell is a committed number with a source file.

| Model | SFT gain (target) | MMLU (0-shot EM) | HellaSwag (acc) | GSM8K (exact match) | TruthfulQA (MC1) | Forgetting Δ |
|---|---|---|---|---|---|---|
| Base (`Llama-3.2-1B`) | — | B₁ | B₂ | B₃ | B₄ | 0 |
| R-SFT (exercise-01 recipe) | +X% | B₁ − d₁ | B₂ − d₂ | B₃ − d₃ | B₄ − d₄ | mean(dᵢ / Bᵢ) |
| M1 (LR 5e-6, 1 epoch) | +X₁% | ... | ... | ... | ... | ... |
| M2 (R-SFT + 30% replay) | +X₂% | ... | ... | ... | ... | ... |

"SFT gain" is a specific target metric defined up front — not loss. Pick one of:

- **Instruction-following**: a 50-prompt panel scored 1–5 by a cheap LLM judge (keep the rubric committed; this is a mod-108 pattern).
- **Format adherence**: a regex/structural checker over a fixed suite of 100 prompts with declared expected formats.
- **Chat loss on `no_robots` test split**: a cheap proxy; defensible if the learner says so.

Whichever is picked, lock it before training.

### 2.2 `eval_matrix.py` — the frozen baseline

The quickest reproducible path is `lm-evaluation-harness` for the general evals and a small bespoke panel for the SFT-gain metric [LMEH]. The harness guarantees the same prompt templates, few-shot exemplars, and scoring that published numbers use.

```python
# eval_matrix.py — single-GPU 1B model; each full matrix run ≈ 20 minutes.
import json, pathlib, subprocess, shlex

MODELS = {
    "base":  "meta-llama/Llama-3.2-1B",
    "r_sft": "runs/sft-1b-v1/final",             # exercise-01 output
    "m1":    "runs/sft-1b-m1-low-lr/final",      # §2.3
    "m2":    "runs/sft-1b-m2-replay/final",      # §2.4
}
TASKS = "mmlu,hellaswag,gsm8k,truthfulqa_mc1"    # lm-eval task names
OUT = pathlib.Path("eval"); OUT.mkdir(exist_ok=True)

for name, path in MODELS.items():
    outdir = OUT / name; outdir.mkdir(exist_ok=True)
    cmd = (
        f"lm_eval --model hf --model_args pretrained={path},dtype=bfloat16 "
        f"--tasks {TASKS} --batch_size 8 --output_path {outdir}"
    )
    subprocess.check_call(shlex.split(cmd))

# summary.json: mean of per-task percent-regressions vs. base.
import statistics as st
base = json.loads(next((OUT/'base').glob('results_*.json')).read_text())["results"]
rows = {}
for name in MODELS:
    r = json.loads(next((OUT/name).glob('results_*.json')).read_text())["results"]
    rows[name] = {
        t: r[t].get("acc,none", r[t].get("exact_match,none"))
        for t in base
    }
    rows[name]["forget_pct"] = st.mean(
        (base[t].get("acc,none", base[t].get("exact_match,none")) -
         rows[name][t]) /
        base[t].get("acc,none", base[t].get("exact_match,none"))
        for t in base
    )
(OUT / "summary.json").write_text(json.dumps(rows, indent=2))
```

Important:

- Dtype must match the training dtype (`bf16`). An eval in fp32 and a train in bf16 produce different numbers; attribute nothing to forgetting until that is held constant.
- Base model is evaluated with its own tokenizer, not the SFT run's saved tokenizer. If the SFT run changed `pad_token` or added a token, the eval harness must reflect that.
- MMLU on a 1B model scores 25–30% — not 60% — because 1B is small. Do not panic at "low" baselines. The question is the *delta*, not the level.

### 2.3 M1 — mitigation 1: lower LR, fewer epochs

Lower LR produces less parameter drift; fewer epochs means less exposure to the SFT distribution. Both cost SFT gain.

```python
# train_m1.py — same as exercise-01's train.py with these overrides:
overrides = dict(
    learning_rate=5e-6,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
)
```

Expected shape: forgetting roughly halves vs. R-SFT. SFT gain drops to roughly 60–80% of R-SFT's gain, depending on which SFT-gain metric was chosen.

Why it works: catastrophic forgetting is a function of integrated update magnitude on the pretraining-distribution weights. Smaller LR × fewer steps = smaller integrated magnitude [LWF] [CF-OVERVIEW].

### 2.4 M2 — mitigation 2: replay (data mixing)

Mix the SFT corpus with a general-capabilities instruction set to keep the model's loss gradient pointed at both targets.

```python
# train_m2.py — mix no_robots with tulu-3-sft-mixture at 30% replay.
from datasets import concatenate_datasets, load_dataset
sft  = load_dataset("HuggingFaceH4/no_robots")["train_sft"]
rep  = load_dataset("allenai/tulu-3-sft-mixture")["train"].shuffle(seed=42)
# Hold replay at ~30% of the final mix; replace ratio with your target.
n_rep = int(len(sft) * (0.30 / 0.70))
mixed = concatenate_datasets([sft, rep.select(range(n_rep))]).shuffle(seed=42)
# Rest of the trainer wiring matches exercise-01 (LR 2e-5, 2 epochs, cosine, 3% warmup).
```

Expected shape: forgetting shrinks further than M1, often below 20% of R-SFT's forgetting, at approximately the same SFT gain as R-SFT. The replay mixture keeps the model's internal representations close to the pretraining manifold while still learning the target format [REHEARSAL] [TULU3].

Caveat: the choice of replay data matters more than the ratio. `tulu-3-sft-mixture` was designed as a general-purpose instruction mix [TULU3]; substituting a narrow domain mix (e.g., medical QA only) will steer forgetting toward whatever that domain *doesn't* cover.

### 2.5 Reading the matrix

Compute a 2-D scatter the learner presents in their write-up:

```
y-axis: SFT gain (relative to Base, same metric chosen above)
x-axis: Forgetting Δ (mean percent regression across MMLU/HellaSwag/GSM8K/TruthfulQA)

Base   ●  (0, 0)
R-SFT  ●  (high x, high y)       — the naive SFT: big gain, big forgetting
M1     ●  (low x, mid y)         — safe but slow learner
M2     ●  (low x, high y)        — the mitigation that actually trades well
```

The report's conclusion is one paragraph: "Given the target metric X, mitigation M2 preserves the base model's general capabilities at Δ=4% while retaining Y% of R-SFT's gain. M1 is the better choice if we are willing to re-train on-demand per customer and lack a general-capabilities replay corpus."

### 2.6 Qualitative probes — the five that make forgetting visible

Numerical deltas are often small because the benchmarks are noisy on 1B models. Qualitative probes catch dramatic forgetting that MMLU misses because the question format doesn't surface it.

Run each on Base, R-SFT, M1, M2 with greedy decoding:

1. "Name three provinces of Canada." *(World knowledge. R-SFT may answer confidently with wrong names.)*
2. "What is 148 × 37? Show the steps." *(Arithmetic. R-SFT may skip steps or hallucinate.)*
3. "Translate 'good evening' into French." *(Cross-lingual retention — pretraining had French; narrow English SFT erodes it.)*
4. "In the text 'I didn't go', what does the contraction expand to?" *(Linguistic competence at the token level — a canary for the tokenizer/model surface drift.)*
5. "Continue this code fragment: `def fibonacci(n):`" *(Code generation. The answer should still be a correct recursive or iterative function.)*

A reviewer should be able to read the four columns of outputs and nod at which model produces which.

### 2.7 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| General benchmarks, not SFT-held-out | Forgetting manifests on pretraining-distribution tasks the SFT data never saw | SFT-held-out loss | SFT-held-out loss mistakes memorization for retention |
| `lm-evaluation-harness` | Published-number parity; sample selection and template fixed | Bespoke eval loop | Bespoke drifts; numbers stop being comparable |
| Freeze baselines first | The only way to attribute later changes to SFT | "Rerun base after SFT for good measure" | "Good measure" reruns on changed cache state, numbers drift ~0.3% |
| LR + epochs for M1 | Single-knob mitigation; isolates "less training" | Lower LR OR fewer epochs alone | Each alone is weaker; combining is the standard practitioner move |
| Replay for M2 | Gives the loss landscape a reason to stay near pretraining | L2 penalty to base weights (EWC-like) | EWC/LWF add machinery; replay is the baseline every mitigation paper compares to |
| Mean-of-regressions, not sum | Normalizes across benchmarks with different absolute scales | Sum of regressions | Sum over-weights GSM8K's larger absolute swings |
| 30% replay ratio | Published mitigation papers cluster 10–50%; 30% is defensible middle | 10% or 50% | Lower → less protection; higher → smaller SFT gain |

## 3. Validation steps

1. **Base frozen.** `git` diff between `eval/base/` and any later re-run must be empty. If the harness emitted different numbers on identical inputs, the harness version drifted or the model path changed.
2. **Same dtype across models.** Every eval command has `dtype=bfloat16` (or whichever was trained). Mixing is a common and silent source of apparent "forgetting."
3. **Tokenizer parity.** Each fine-tuned model's `tokenizer.model_max_length`, special tokens, and chat template match the base's. Any change means the delta is partly tokenizer drift.
4. **Benchmarks are disjoint from SFT.** Run a cheap contamination check (n-gram overlap) between `no_robots` training rows and MMLU questions. Any match → the benchmark is compromised for this experiment; swap to another.
5. **Replay ratio holds through training.** Sample 100 batches during M2 training; confirm ~30% of examples trace back to the replay dataset. Shuffle + mix bugs silently drop the replay ratio to zero.
6. **Qualitative probes committed.** The five-prompt panel outputs land in `eval/qualitative.md`. Without a committed read, the matrix is just numbers.
7. **Seed stability.** Rerun M2 at a second seed. Forgetting Δ changes by < 1 percentage point. Larger swings mean the mitigation's effect is in the noise band and the conclusion is premature.

## 4. Rubric for grading the learner's investigation

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Frozen baseline** | Base evaluated after SFT, or no base numbers | Base numbers present but one eval re-ran | Base evaluated first, committed, never re-touched; matched dtype/tokenizer across all evals |
| **General eval coverage** | ≤1 benchmark, or an SFT-overlapping benchmark | ≥2 benchmarks but no contamination check | MMLU + HellaSwag + GSM8K + TruthfulQA (or equivalent), contamination check done |
| **Two mitigations run** | 0 or 1 mitigation | 2 mitigations but on different base/data | Exactly 2 mitigations against the same base + target, one knob per mitigation |
| **Trade-off framing** | Claims a "best" model without naming the exchange rate | States the trade-off but doesn't quantify it | Reports forgetting Δ and SFT gain side by side; names which mitigation is better under which operating assumption |
| **Qualitative probes** | None, or probes only on the SFT model | Probes on 2 models | Probes on all four, output table committed, at least one named regression |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Measuring "forgetting" on a held-out slice of the SFT corpus. Held-out SFT measures memorization, not forgetting. The regression sits on the pretraining distribution.
- Re-running the base model's benchmarks after SFT "for completeness." The cache, the tokenizer save state, or the `transformers` version has probably drifted; the delta is now contaminated.
- Dtype mismatch between train and eval. A model trained bf16 and evaluated fp32 scores systematically different; the diff is dtype, not forgetting.
- Choosing a replay dataset that overlaps with the general benchmarks (e.g., mixing in MMLU-style practice data). Forgetting Δ now reads as "mitigated" because the model saw the test distribution.
- Concluding "M2 wins" without a committed SFT-gain metric. Without it, the comparison is forgetting-only and the mitigation that does nothing (base) always wins on that axis.
- Treating LoRA as a forgetting mitigation without saying so. LoRA constrains the update to a low-rank subspace, which reduces forgetting as a side effect [LORA]; the experiment belongs to mod-104 but not naming the link leaves a gap.
- Reporting a single metric number, not a confidence band. MMLU on 1B swings 0.3–0.5 absolute points by seed; a 0.4-point "regression" is noise.
- Skipping the qualitative probes. Benchmarks under-weight world-knowledge collapse; the probes are the clearest way to show it exists.

## 6. References

- [L3] Meta, "The Llama 3 Herd of Models," 2024. <https://arxiv.org/abs/2407.21783>
- [NOROBOTS] HuggingFace H4, "No Robots" dataset. <https://huggingface.co/datasets/HuggingFaceH4/no_robots>
- [TULU3] Lambert et al., "Tulu 3: Pushing Frontiers in Open Language Model Post-Training," AI2, 2024 — `allenai/tulu-3-sft-mixture`. <https://huggingface.co/datasets/allenai/tulu-3-sft-mixture>
- [MMLU] Hendrycks et al., "Measuring Massive Multitask Language Understanding," 2021. <https://arxiv.org/abs/2009.03300>
- [HELLA] Zellers et al., "HellaSwag: Can a Machine Really Finish Your Sentence?" 2019. <https://arxiv.org/abs/1905.07830>
- [GSM8K] Cobbe et al., "Training Verifiers to Solve Math Word Problems," 2021. <https://arxiv.org/abs/2110.14168>
- [TRUTHFULQA] Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods," 2022. <https://arxiv.org/abs/2109.07958>
- [LMEH] EleutherAI, `lm-evaluation-harness`. <https://github.com/EleutherAI/lm-evaluation-harness>
- [CF-OVERVIEW] French, "Catastrophic Forgetting in Connectionist Networks," 1999. <https://www.sciencedirect.com/science/article/abs/pii/S1364661399013942>
- [LWF] Li & Hoiem, "Learning without Forgetting," ECCV 2016. <https://arxiv.org/abs/1606.09282>
- [REHEARSAL] Robins, "Catastrophic Forgetting, Rehearsal and Pseudorehearsal," 1995. <https://www.tandfonline.com/doi/abs/10.1080/09540099550039318>
- [LORA] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021 — cited here as a forgetting-mitigation side effect; full treatment in mod-104. <https://arxiv.org/abs/2106.09685>
- OWASP Machine Learning Security Top 10 — ML-02 Data Poisoning (a biased replay mix is poisoning by omission). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — "Erode ML Model Integrity" category of impact. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework — Measure 2.5 (performance and robustness); Manage 3 (ongoing monitoring and mitigation). <https://www.nist.gov/itl/ai-risk-management-framework>
