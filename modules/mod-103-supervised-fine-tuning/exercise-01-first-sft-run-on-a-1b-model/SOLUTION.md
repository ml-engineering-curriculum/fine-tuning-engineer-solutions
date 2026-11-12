# mod-103 · exercise-01 — First SFT Run on a 1B Model — Solution

## 1. Solution overview

The exercise is not a race to the lowest loss. It is the first time the learner wires every moving part of an SFT run end-to-end: a small open-weights base model, a chat-templated dataset with assistant-only labels, a `Trainer` configured for a single-GPU box, a committed `training_report.md`, and a before/after sanity generation that a reader can diff.

Three invariants carry the exercise:

- **Base model and tokenizer stay in lockstep.** The tokenizer loaded into the trainer must be the tokenizer the base model was trained with. A mismatch silently shifts every BOS, special-token, and chat-template boundary and produces loss curves that look fine while generating garbage.
- **Labels are assistant-only, terminator included.** Training on the user turn teaches the model to speak the user's part; training without the terminator teaches it to over-generate. TRL's `SFTTrainer` with `assistant_only_loss=True` (chat-templated datasets) is the default correct choice [TRL].
- **Every number in the training report has a pointer.** Loss, step count, tokens-seen, hardware, seed, dataset hash, git SHA — each one must be derivable from a committed file. "I got 1.42 loss" without the config to reproduce it is not a result.

The worked walkthrough uses `meta-llama/Llama-3.2-1B` as the base and `HuggingFaceH4/no_robots` as the SFT corpus [L3] [NOROBOTS]. The structure is unchanged for `Qwen/Qwen2.5-0.5B` or `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`; only the chat-template and context-length numbers shift.

## 2. Worked implementation

### 2.1 Choices and justification

| Choice | Value | Why |
|---|---|---|
| Base model | `meta-llama/Llama-3.2-1B` | Open-weights, 1B params, fits on a 24 GB GPU at bf16 full-parameter SFT with batch-size 1 and `gradient_checkpointing` [L3] |
| Dataset | `HuggingFaceH4/no_robots` (9.5k rows) | Permissively licensed, chat-templated, small enough to finish in <1 hour on a single consumer GPU [NOROBOTS] |
| Format | HuggingFace `messages` schema | Lets `SFTTrainer` apply the model's chat template automatically [TRL] |
| Trainer | `trl.SFTTrainer` | Owns assistant-only masking, packing, and chat-template rendering in one class [TRL] |
| Precision | bf16 (A100/H100/RTX 30+) or fp16 (older) | bf16 avoids the fp16 dynamic-range overflow that destabilizes the first 1–2% of training |
| Seq length | 2048 | Covers >99% of `no_robots` rows at the Llama-3 tokenizer; drops the rest |
| Batch | `per_device_train_batch_size=2`, `gradient_accumulation_steps=16` | Effective batch 32 at seq-len 2048 ≈ 65k tokens/step — within the 50–100k sweet spot for 1B models |
| LR | 2e-5 | Mid of the 1e-5 to 5e-5 range common for full-parameter small-model SFT on short runs |
| Schedule | `cosine`, `warmup_ratio=0.03` | Warmup prevents the early-step spike; cosine's slow tail protects against the end-of-run overfitting bump |
| Epochs | 2 | `no_robots` is small; 1 epoch under-fits the format, 3+ starts to parrot |
| Seed | `42` everywhere (`torch`, `numpy`, `random`, `transformers.set_seed`) | Reproducibility; see exercise-05 |

### 2.2 `train.py` (reference)

```python
# train.py
from __future__ import annotations
import os, json, hashlib, pathlib, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, set_seed,
)
from trl import SFTTrainer, SFTConfig

MODEL_ID = "meta-llama/Llama-3.2-1B"
DATA_ID  = "HuggingFaceH4/no_robots"
OUT      = pathlib.Path(os.environ.get("OUT", "runs/sft-1b-v1"))
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
set_seed(SEED); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.use_deterministic_algorithms(False)  # perf; exercise-05 flips this on for the audit run

def load_data():
    ds = load_dataset(DATA_ID)
    # no_robots ships with a "messages" column already in the HF chat schema.
    # Drop rows with empty messages or non-assistant-terminal tails; those are noise.
    def ok(ex):
        m = ex["messages"]
        return bool(m) and m[-1]["role"] == "assistant" and all(x["content"].strip() for x in m)
    return ds.filter(ok)

def sha256_of_rows(ds):
    h = hashlib.sha256()
    for ex in ds:
        h.update(json.dumps(ex["messages"], sort_keys=True).encode())
    return h.hexdigest()

def main():
    ds = load_data()
    train_ds = ds["train_sft"] if "train_sft" in ds else ds["train"]
    eval_ds  = ds["test_sft"]  if "test_sft"  in ds else ds["test"]

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        # Llama 3 ships without a pad token; reserve one of the special tokens.
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)

    cfg = SFTConfig(
        output_dir=str(OUT),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=200, save_total_limit=3,
        report_to=["none"],  # tensorboard/wandb is fine; pin it if you turn it on
        seed=SEED, data_seed=SEED,
        # SFT-specific:
        max_seq_length=2048,
        packing=False,                    # covered in exercise-02
        assistant_only_loss=True,         # labels mask all non-assistant tokens
        dataset_text_field=None,          # use messages column
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Record the exact inputs *before* training so a failed run still has a receipt.
    (OUT / "dataset_sha256.txt").write_text(sha256_of_rows(train_ds) + "\n")
    (OUT / "model_id.txt").write_text(MODEL_ID + "\n")
    (OUT / "config.json").write_text(cfg.to_json_string())

    result = trainer.train()
    trainer.save_model(str(OUT / "final"))
    tok.save_pretrained(str(OUT / "final"))
    (OUT / "train_result.json").write_text(json.dumps(result.metrics, indent=2))

if __name__ == "__main__":
    main()
```

Notes:

- `SFTTrainer` with `assistant_only_loss=True` and a `messages`-schema dataset renders the chat template, builds `input_ids`, and masks every non-assistant token to `-100` without a custom collator [TRL].
- `attn_implementation="sdpa"` is the PyTorch 2 scaled-dot-product path. FlashAttention-2 is a drop-in improvement where the hardware supports it; leave it off for the first run so a subtle FA bug does not get attributed to the SFT recipe.
- `packing=False` here. Packing is the subject of exercise-02; mixing it with a first run obscures which change caused which loss behavior.
- `gradient_checkpointing_kwargs={"use_reentrant": False}` silences a `torch.utils.checkpoint` deprecation and keeps the backward pass non-reentrant (no state hidden across calls).

### 2.3 `smoke_generate.py` — before/after qualitative check

```python
# smoke_generate.py — usage: python smoke_generate.py BASE_OR_FINETUNED_DIR
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Write a two-sentence email asking my landlord when the elevator will be fixed.",
    "Explain gradient accumulation to a backend engineer in three sentences.",
    "Give me a Python one-liner that reads a JSONL file and yields each row as a dict.",
    "I have a poem that is almost good: 'Roses are red, violets are blue'. Finish it in two more lines.",
    "What time is it in Tokyo right now?",  # expected: honest refusal of real-time info
]

def main(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for p in PROMPTS:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False,
                                 temperature=0.0, repetition_penalty=1.05,
                                 eos_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"=== {p}\n{text}\n")

if __name__ == "__main__":
    main(sys.argv[1])
```

Expected observations:

- **Base model** (no instruct variant): the `apply_chat_template` call still works because Llama 3.2 ships a chat template on the base tokenizer, but the base model mostly continues the prompt in document style — paragraphs rambling past the ask, no sensible stop.
- **SFT model (2 epochs, 2e-5)**: short, on-task answers; stops at the assistant turn terminator; refuses the real-time question rather than inventing a clock reading.

Generation differences this stark are expected because the base model never saw `<|eot_id|>` as a "stop here" signal in its pretraining mixture at scale. SFT teaches it that stop. If the SFT model still runs on past the question, labels did not include the terminator — re-check §2.2's `assistant_only_loss`.

### 2.4 `training_report.md` — target shape

```markdown
# SFT Run — Llama-3.2-1B on no_robots (v1)

**Run id:** `sft-1b-v1-2026-11-11`
**Git SHA:** `a1b2c3d` (dirty: no)
**Start:** 2026-11-11 14:02 UTC  **End:** 2026-11-11 14:47 UTC

## Inputs

| What | Value |
|---|---|
| Base model | meta-llama/Llama-3.2-1B (rev `main` @ 2024-09-25) |
| Dataset | HuggingFaceH4/no_robots `train_sft` — 9,500 rows, sha256 `e3f0...9c12` |
| Eval split | HuggingFaceH4/no_robots `test_sft` — 500 rows |
| Tokenizer | base-model tokenizer, Llama-3 chat template |
| Max seq len | 2048 (dropped 41 train rows, 2 eval rows) |

## Compute

| What | Value |
|---|---|
| GPU | 1x NVIDIA RTX 4090 (24 GB) |
| Precision | bf16 |
| `attn_implementation` | sdpa |
| `gradient_checkpointing` | on, use_reentrant=False |
| Effective batch | 2 × 16 = 32 |
| Approx tokens/step | 65,536 |
| Total steps | 594 |
| Wall clock | 45m 11s |

## Loss curve

- Train loss: 1.78 → 1.41 → 1.23 → 1.12 (steps 0, 150, 300, 594).
- Eval loss: 1.69 → 1.38 → 1.19 → 1.10 (steps 100, 200, 400, 594). Monotonic decrease; no late-run bump.
- Grad norm stable in [0.4, 1.1] after step 20.
- No NaN/Inf events logged.

## Qualitative

See `smoke_generate.base.txt` and `smoke_generate.final.txt`. Base rambled on 4 of 5 prompts; final answered on-task on all 5 and refused the real-time clock question without inventing a time.

## Known limitations

1. Single-seed run — variance not characterized. Exercise-05 does a 3-seed re-run and reports the spread.
2. Full-parameter SFT on 1B; LoRA/QLoRA comparison lives in mod-104.
3. No contamination check against general eval benchmarks — mod-102 exercise-02 covers that.

## Reproduce

```
git checkout a1b2c3d
pip install -r requirements.txt
OUT=runs/sft-1b-v1 python train.py
python smoke_generate.py runs/sft-1b-v1/final
```
```

### 2.5 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Full-parameter SFT (not LoRA) | First run teaches the baseline; LoRA then has something to compare against | QLoRA at 4-bit | Lower peak memory; extra moving parts (bnb, PEFT adapters) hide bugs on the first run |
| 2 epochs | `no_robots` is 9.5k rows; 1 epoch under-fits the chat format, 3+ starts to echo | 1 or 3 | 1 → format not learned; 3 → memorization |
| bf16 over fp16 | fp16's narrow dynamic range causes the early-step spike that eats half of beginners' first runs | fp16 | fp16 is faster on pre-Ampere; keep it for compute-bound pretraining, not SFT recipes |
| `assistant_only_loss=True` | Train on what the model should emit; see mod-102 exercise-03 | Mask nothing (train on everything) | Train-on-everything wastes capacity on user turns and over-generates |
| 1B base | Fits a single consumer GPU; short enough wall clock to iterate | 3B, 7B | Larger model needs more VRAM and more careful LR; not what this exercise is for |
| Eval every 100 steps | Enough signal to notice overfitting mid-run without costing too much | Every epoch | Epoch-level eval misses the mid-run inflection; eval-every-10 wastes time |

## 3. Validation steps

1. **Tokenizer/model agreement.** `AutoTokenizer.from_pretrained(MODEL_ID).vocab_size` must equal `model.config.vocab_size`. If not, embeddings and tokens are misaligned — common after swapping tokenizers.
2. **Pad is distinct from loss.** `tok.pad_token_id != tok.eos_token_id` OR the collator masks pads to `-100`. With `assistant_only_loss`, pads inside the system/user turns are already `-100`, but post-sequence pads at the end of a packed sequence are not — be explicit.
3. **Label mask spot-check.** On ten random training rows, assert that every position where `labels == -100` decodes to a system/user token and every position where `labels != -100` decodes to assistant content or its terminator.
4. **Terminator in loss.** For each training row, `labels[-1] == input_ids[-1]` (the terminator token id, e.g. `<|eot_id|>` = 128009 for Llama 3) rather than `-100`.
5. **Loss sanity on step 1.** The base is already a pretrained LM, so the opening step-1 train loss is the model's natural next-token loss on assistant tokens rendered through its own chat template — typically 1.5–3.0 for an English chat corpus on Llama-3.2-1B (this run reports 1.78; see §2.4). `ln(vocab_size)` ≈ 11.76 is the uniform-prediction ceiling, not the expected opening loss, and would only apply to a randomly-initialised model. Treat as suspicious: an opening loss *above* ~6 (suggests the tokenizer or chat template does not match the base, so assistant tokens are being predicted from near-uniform logits), or an opening loss *below* ~0.3 (suggests the assistant tokens are nearly-perfectly predicted — either the base already saw this corpus, or the label mask has collapsed to just the EOS token).
6. **No grad-norm explosion.** `grad_norm` logged at each step stays below 10 after step ≈ 20. A single spike above 100 followed by NaN loss is the fp16-range failure; the fix is bf16 or `max_grad_norm` enforcement.
7. **Before/after sanity.** Run `smoke_generate.py` against both the base dir and the `final/` dir. The base should ramble; the SFT model should stop. Save both outputs in the report.
8. **Resume works.** Let training write at least one checkpoint first — `save_steps=200` means the first `checkpoint-200/` directory is not on disk until step 200 completes; killing earlier leaves nothing to restore. After `checkpoint-200/` is present, kill the process and relaunch with `trainer.train(resume_from_checkpoint=True)` (or pass the explicit checkpoint path). Verify that step counter resumes from 200, loss continues the curve instead of restarting, and the optimizer `m`/`v` states load (no first-step grad-norm spike on the restart). If resume restarts from step 0 despite the resume flag, the checkpointing is writing model weights but not optimizer state.

## 4. Rubric for grading the learner's run

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Config completeness** | Missing one of model/dataset/seed/LR/schedule/batch/precision | All present but at least one value is a framework default pasted from a tutorial | All present and each value has a one-line "why" in the report |
| **Correct loss masking** | Trains on everything, or masks assistant content | Masks user turns but excludes the assistant terminator from loss | `assistant_only_loss=True` (or equivalent), terminator included in loss, verified on a spot-check |
| **Loss curve read** | Only a final-loss number reported | Full curve present but no interpretation | Train + eval loss plotted/tabulated, slope-change points named, grad-norm included |
| **Before/after sanity** | Only final-model generation | Both base and final generations shown but not diffed | Both shown, diff described, at least one case where behavior changed is called out |
| **Training report** | Missing, or counts do not reconcile with saved artifacts | Report present but missing git SHA, dataset hash, or reproduce block | Complete report: inputs with hashes, compute block, loss curve, limitations, reproduce command |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Setting `pad_token = eos_token` and *not* masking pads — the model learns to emit end-of-text every few tokens because the loss rewards it.
- Using the instruct variant (`Llama-3.2-1B-Instruct`) as the base. The model already speaks chat; the exercise's before/after signal collapses.
- Keeping `max_seq_length` at the model's native context (131k on Llama 3.2) when the dataset lives under 1k tokens. Wastes VRAM, slows training by orders of magnitude, no quality win.
- Training with `fp16` on first run. The opening-step spike is diagnostic noise that eats debugging time; bf16 removes it.
- Logging only the final loss. The first 2% of training is where most bugs live; without a curve, you cannot see them.
- Omitting the dataset hash and the git SHA. The run is unreproducible even to yourself a week later.
- Reporting `train_loss` as "the result" without an eval split. Pure training loss on SFT is a red flag — it always goes down.
- Enabling `packing=True` on the first run and then wondering why the loss curve looks different from every tutorial. Packing changes effective tokens-per-step; introduce it on its own in exercise-02.
- Skipping `gradient_checkpointing` on a 24 GB GPU and OOM'ing at step 1, then dropping batch to 1 without re-tuning gradient accumulation — effective batch silently changes and the LR is now mis-scaled.

## 6. References

- [L3] Meta, "Introducing Llama 3.2" (model cards + license), 2024. <https://www.llama.com/> / HF: <https://huggingface.co/meta-llama/Llama-3.2-1B>
- [NOROBOTS] HuggingFace H4, "No Robots" dataset. <https://huggingface.co/datasets/HuggingFaceH4/no_robots>
- [TRL] HuggingFace TRL — `SFTTrainer` and `SFTConfig` documentation. <https://huggingface.co/docs/trl/main/en/sft_trainer>
- [HFTRAIN] HuggingFace Transformers `Trainer` / `TrainingArguments`. <https://huggingface.co/docs/transformers/main/en/trainer>
- [HFTPL] HuggingFace chat templating (`apply_chat_template`, `return_assistant_tokens_mask`). <https://huggingface.co/docs/transformers/main/en/chat_templating>
- [BF16] Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training," 2019. <https://arxiv.org/abs/1905.12322>
- OWASP Machine Learning Security Top 10 — ML-02 Data Poisoning, ML-06 AI Supply Chain Vulnerabilities. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — "Poison Training Data" and "ML Supply Chain Compromise" techniques. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Govern/Map/Measure/Manage; model and data provenance obligations). <https://www.nist.gov/itl/ai-risk-management-framework>
