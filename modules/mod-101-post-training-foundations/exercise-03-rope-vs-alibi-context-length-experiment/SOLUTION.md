# mod-101 · exercise-03 — RoPE vs ALiBi Context-Length Experiment — Solution

## 1. Solution overview

The exercise asks the fine-tuner to design (and ideally run) an experiment comparing two positional encoding schemes — **RoPE** (Rotary Position Embeddings) [RoPE22] and **ALiBi** (Attention with Linear Biases) [ALiBi21] — under the lens that matters most to a post-training engineer: **what happens to quality when the eval context length is longer than the training context length?**

The interesting answer is subtle:

- **ALiBi** was designed to extrapolate out-of-the-box. It biases attention logits by `−m · |i−j|` (slope `m` per head, fixed by the architecture, not learned). A model trained at 1024 tokens with ALiBi can decode at 2048+ with modest degradation [ALiBi21].
- **RoPE** does *not* extrapolate zero-shot past its training window — logit magnitudes blow up as position indices exceed trained ranges [RoPE22, PI23]. But RoPE *does* admit cheap, effective context-length extension via **position interpolation (PI)** [PI23], **NTK-aware scaling** [NTK23], and **YaRN** [YaRN23]. With those methods, RoPE-based models now dominate long-context recipes (Llama 3.1 at 128K, Qwen 2.5 at 128K, etc.) [L3].
- Modern deployed decoder-only LLMs almost universally use RoPE, not ALiBi. The practical question is therefore not "which to pick for a fresh training run" (RoPE + a scaling method is the default) but "how do we evaluate extension faithfully?"

This solution gives (1) a worked experiment design, (2) a decision matrix the student should fill in, (3) the evaluation protocol, (4) a rubric, and (5) the common mistakes that invalidate the comparison.

## 2. Implementation — worked experiment design

### 2.1 The question, pinned down

Pick exactly one of these three questions — they imply different experiments:

- **Q-FROM-SCRATCH.** "If I pretrain from scratch at 2K context, which positional scheme has better perplexity at 2K, 4K, 8K, and 16K at eval time?" This is the original ALiBi paper's framing [ALiBi21].
- **Q-EXTENSION.** "Given a RoPE-pretrained base at 8K context, which context-extension method (none / linear PI / NTK-aware / YaRN) best preserves quality at 32K and 128K?" This is the practical fine-tuning framing.
- **Q-RETROFIT.** "Can I retrofit an existing RoPE model to ALiBi (or vice versa) with light fine-tuning?" This is almost always a bad idea; included for completeness.

For a learner-budget exercise, **Q-EXTENSION** is the recommended question: it is the one a fine-tuning engineer actually faces, RoPE is already the default, and all four arms are cheap to configure (no retraining required to produce extended models; only a configuration + short continued-pretrain or SFT pass).

The rest of this solution answers **Q-EXTENSION**. The design generalizes to Q-FROM-SCRATCH by substituting two small pretrained-from-scratch models.

### 2.2 The design matrix

| Arm | Position encoding | Training ctx | Eval ctx (held out) | Extension method | Expected behavior |
|---|---|---|---|---|---|
| A | RoPE | 8K | 8K / 32K / 128K | none (just extend `max_position_embeddings`) | OK at 8K; degrades rapidly past 8K |
| B | RoPE | 8K | 8K / 32K / 128K | Linear PI (scale `m` by L_train/L_target) | OK at trained target; slight quality loss |
| C | RoPE | 8K | 8K / 32K / 128K | NTK-aware (scale `θ_base`, not `m`) | OK; preserves high-freq channels better than PI |
| D | RoPE | 8K | 8K / 32K / 128K | YaRN | Generally best among RoPE scaling methods per paper [YaRN23] |
| E | ALiBi | 8K | 8K / 32K / 128K | none | Extrapolates per [ALiBi21]; degrades more slowly than A |

At minimum, pick arms **A, B, C** and (if time) **D** and **E**. Arms B–D need only a config change *plus* a short continued pretrain pass on long-context documents to let the model adapt to the new effective positions (typically 100M–1B tokens per published recipes [L3, YaRN23]). A learner on a laptop can skip the continued-pretrain pass but must then interpret results strictly as "zero-shot scaling", not "fine-tuned long context."

### 2.3 Base model and eval data

- **Base.** A RoPE-based open model in the 1B–8B range with a well-documented `rope_scaling` config: Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B. For Arm E, a public ALiBi model (e.g. MPT-7B) — or pretraining a small LM from scratch if the learner has the budget.
- **Eval data.** Two complementary eval families:
  - **Language modeling PPL** on long documents (books, long papers). Measure perplexity of tokens at positions 1K, 4K, 16K, 64K to see degradation as a function of position, not just average over the window.
  - **Needle-in-a-haystack retrieval** [NIH23] at depths 10%, 25%, 50%, 75%, 90% of the context window. This isolates *retrieval* from local-fluency perplexity; a model can have reasonable PPL at long context while still failing to locate a fact inserted mid-document.
- Hold out the eval set; do not train on it.

### 2.4 Decision rationale (what to defend)

The student is expected to defend:

1. **Why Q-EXTENSION, not Q-FROM-SCRATCH?** Because the fleet of production LLMs is RoPE-based; the engineer rarely gets to pick positional scheme from scratch. The decision that matters is which extension method to apply.
2. **Why multiple eval context lengths, not just the max?** A model may have good PPL at 128K-averaged while being much worse than baseline at 16K — a hidden regression that only mid-length evals catch [YaRN23].
3. **Why both PPL and needle-in-haystack?** PPL is a local metric; needle tests long-range dependency. They measure different failure modes [NIH23].
4. **Why pin the random seed, batch size, and tokenizer across arms?** So the only variable is the positional encoding or scaling factor. Changing tokenizer between arms makes PPL non-comparable.
5. **Why hold `θ_base` in the reported config?** Because NTK-aware and YaRN both modify it, and reproducibility requires the exact post-scaling value.

### 2.5 Protocol (step by step)

1. Pick base, fix tokenizer, fix eval set, fix sampling seed.
2. For each arm:
   - Load base with the arm's `rope_scaling` config (HF Transformers: `model.config.rope_scaling = {"type": "linear", "factor": 4.0}` for PI × 4; `"type": "dynamic"` for NTK-aware; `"type": "yarn"` for YaRN) [HF-ROPE].
   - (Optional but strongly recommended) Continued-pretrain on long-document corpus at the new target length for N tokens; stop when long-ctx PPL plateaus.
   - Evaluate PPL at positions {1K, 4K, 16K, 64K} on the long-doc eval set.
   - Evaluate needle-in-haystack at depths {10, 25, 50, 75, 90}% at target contexts {8K, 32K, 128K}.
3. Report a table: arm × eval-length × metric.
4. Compute and report:
   - Degradation slope (PPL at position `p` − PPL at position 1K).
   - Retrieval accuracy heatmap (depth × context length) per arm.
5. Choose the arm that minimizes PPL slope **and** keeps retrieval ≥ 90% at the target production context.

### 2.6 Minimal reproducible snippet

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID    = "<a RoPE-based open base, e.g. Llama 3.1 8B>"
TARGET_CTX  = 32768

cfg = AutoConfig.from_pretrained(MODEL_ID)
# Arm B — linear position interpolation ×4
cfg.rope_scaling = {"type": "linear", "factor": TARGET_CTX / cfg.max_position_embeddings}
cfg.max_position_embeddings = TARGET_CTX

tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg,
                                             torch_dtype="auto")

# (Optional) continued pretrain on long documents here.

# Positional PPL evaluation
import torch
def ppl_at_positions(model, tok, text, positions, window):
    ids  = tok(text, return_tensors="pt").input_ids.cuda()
    ids  = ids[:, :window]
    with torch.no_grad():
        out = model(ids, labels=ids)
    # Per-position NLL; average in a ±128-token window around each target position.
    shift_logits = out.logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(1, -1)
    return {p: nll[0, max(0, p-128):p+128].mean().exp().item() for p in positions}
```

The learner then runs the same harness across arms, swapping only `cfg.rope_scaling` (or replacing the base model for arm E).

## 3. Validation steps

A valid report answers "yes" to each:

1. **Arms share tokenizer and eval seed.** Document the exact tokenizer id and eval file hashes.
2. **Eval context lengths include intermediate points, not just the max.** `{8K, 32K, 128K}` or similar.
3. **Both PPL and retrieval evals reported.** A single-metric report can hide regressions [NIH23].
4. **Positional PPL, not just whole-window PPL.** Mean over window hides where in the window quality falls apart.
5. **Scaling configs recorded exactly.** `rope_scaling.type`, `factor`, final `θ_base`, final `max_position_embeddings`.
6. **Zero-shot vs continued-pretrain clearly labeled.** Mixing those two regimes invalidates the comparison.
7. **Comparison includes a baseline that does nothing** (Arm A). Without a "no scaling" baseline, the student cannot show that scaling helped.
8. **Statement of the decision.** One arm is picked, with a one-paragraph defense grounded in the reported numbers — not a vibes-based "YaRN seemed best."

## 4. Rubric

10 points total.

| Dimension | Pts | Pass criterion |
|---|---|---|
| Question pinned | 1 | Exactly one of Q-FROM-SCRATCH / Q-EXTENSION / Q-RETROFIT, with justification |
| Design matrix | 2 | ≥3 arms; variables held constant across arms; extension methods correctly named |
| Eval protocol | 2 | Multiple eval context lengths; PPL + retrieval; eval data isolated from training |
| Config reproducibility | 1 | Full `rope_scaling` / `max_position_embeddings` / `θ_base` recorded per arm |
| Zero-shot vs continued-pretrain | 1 | Labeled clearly; not conflated |
| Results table + slope metric | 2 | PPL(position) reported; retrieval heatmap or table reported |
| Decision + rationale | 1 | Picked arm defended with numbers, not vibes |

Pass threshold: 7/10, with "eval protocol" ≥ 1.

## 5. Common mistakes

- **Reporting only final-position or window-mean PPL.** Hides mid-window regressions; the whole reason to measure at multiple positions is that scaling methods trade quality differently along the window [YaRN23].
- **Comparing models with different tokenizers.** PPL is not comparable across tokenizers.
- **Running the comparison purely zero-shot and concluding "YaRN is the best scaling method."** Zero-shot YaRN beats zero-shot PI, but the published recipes assume continued pretraining [YaRN23]. Call out which regime you are measuring.
- **Trusting `max_position_embeddings` without checking `rope_scaling`.** HF models expose both; `rope_scaling` is what actually changes the positional behavior.
- **Omitting an unscaled baseline (Arm A).** Any scaling method will look like an improvement over vacuum; the question is whether it beats "do nothing but raise the position cap."
- **Using needle-in-haystack as the only retrieval metric.** It is fluent but narrow; pair with at least one multi-hop or document QA eval for a release-level claim [NIH23].
- **Claiming ALiBi "generalizes better than RoPE" based on 2021 numbers alone.** The 2021 result holds vs. unscaled RoPE; modern RoPE + YaRN closes or reverses the gap [YaRN23, L3]. Defend claims against *current* RoPE, not raw-RoPE.
- **Retrofitting ALiBi onto a RoPE base via a few steps of fine-tuning.** The positional geometry the model's QKV has learned is incompatible; expect large quality loss.

## 6. References

- [AIAYN] Vaswani et al., "Attention Is All You Need," 2017. <https://arxiv.org/abs/1706.03762>
- [RoPE22] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2022. <https://arxiv.org/abs/2104.09864>
- [ALiBi21] Press, Smith, Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation," ICLR 2022. <https://arxiv.org/abs/2108.12409>
- [PI23] Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation," 2023. <https://arxiv.org/abs/2306.15595>
- [NTK23] bloc97, "NTK-Aware Scaled RoPE" (community write-up of the NTK-aware approach; no formal arXiv paper — the community write-up is the de facto canonical reference, and YaRN [YaRN23] cites it as such). <https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/>
- [YaRN23] Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models," 2023. <https://arxiv.org/abs/2309.00071>
- [L3] Meta, "The Llama 3 Herd of Models," 2024. <https://arxiv.org/abs/2407.21783>
- [NIH23] Kamradt, "Needle In A Haystack — Pressure Testing LLMs" (methodology, open source). <https://github.com/gkamradt/LLMTest_NeedleInAHaystack>
- [HF-ROPE] HuggingFace Transformers, RoPE scaling configuration docs. <https://huggingface.co/docs/transformers/main/en/main_classes/configuration>
