# mod-104 · exercise-04 — PEFT Methods Comparison Matrix — Solution

## 1. Solution overview

This is a documentation exercise. No training runs are required for the deliverable itself; the output is a comparison matrix across major PEFT methods and a short decision guide that picks a default for each common scenario (single-task SFT, multi-tenant, instruction tuning on a quantized base, prompt-shaped tasks). A reviewer should be able to read `peft_matrix.md` and `method_decisions.md` and pick a method for a new problem without reading any paper.

Two ideas carry the exercise:

- **"PEFT" is a family, not a method.** The methods differ along four axes — where the delta lives (weight-space vs. activation-space vs. input-space), what dtype it's in, how much it costs at inference, and what it serves well. Pick one axis at a time to decide.
- **Hype decays faster than the matrix.** Papers claim 10% quality bumps that disappear on a different eval and reappear on another. The matrix stays anchored to numbers you can reproduce: parameter count, VRAM footprint, inference-latency delta vs. base, and library support.

Scope: LoRA, QLoRA, DoRA, AdaLoRA, Prefix Tuning, Prompt Tuning, P-Tuning v2, Adapters (Houlsby / Pfeiffer), IA³, BitFit, and full fine-tuning as the baseline. Soft-prompt variants and reparameterized methods are called out but not each enumerated.

## 2. Worked implementation

### 2.1 `peft_matrix.md` — the main artifact

```markdown
# PEFT Methods — Comparison Matrix

Reference base for all numbers below: a 7B decoder-only LLM with
d_model=4096, n_layers=32, 32 heads, SwiGLU MLP, GQA (n_kv_heads=8). Numbers
are order-of-magnitude for typical r/bottleneck choices — exact values move
with hyperparameters.

## A. Parameter-space methods (modify weights or weight-equivalents)

| Method | Where the delta lives | Typical trainable params (7B base) | % of base | Train-time add'l VRAM | Inference overhead | Library support |
|---|---|---:|---:|---|---|---|
| Full FT | All weights | ~7.0 B | 100% | ~14 GB for optimizer state (fp32 AdamW) | 0 | All |
| **LoRA** [LoRA21] | Low-rank delta A·B on selected `nn.Linear` | ~20–45 M (r=8–16 on qkvo±mlp) | 0.3–0.6% | ~0.1 GB grads + ~0.1 GB adamw state | ~3% (wrapped forward) or 0% if merged | `peft`, vLLM, LoRAX |
| **QLoRA** [QLoRA23] | LoRA on top of NF4-quantized frozen base | Same as LoRA | 0.3–0.6% | Same as LoRA, base ~4× smaller | Base dequant adds ~3–7% decode latency | `peft` + `bitsandbytes` |
| **DoRA** [DoRA24] | Decompose weight into magnitude × direction; LoRA on direction | ~25–50 M | 0.4–0.7% | ~10% over LoRA | ~5% (two-path forward) or 0% if merged | `peft` |
| **AdaLoRA** [AdaLoRA23] | LoRA with SVD-based adaptive rank per layer | Varies with budget; ~0.3–0.6% | 0.3–0.6% | Same as LoRA + singular-value tracking | ~3% unmerged | `peft` |
| **Adapters (Houlsby)** [Houlsby19] | Bottleneck MLP inserted after attn and after MLP, in series | ~70–100 M (bottleneck=64) | ~1.0% | ~0.2 GB | ~10–15% unavoidable (serial insertion) | `adapters` (AdapterHub) |
| **Adapters (Pfeiffer)** [Pfeiffer20] | Single bottleneck after MLP | ~40–60 M | ~0.6% | ~0.15 GB | ~6–10% unavoidable | `adapters` |
| **IA³** [IA3-22] | Learned per-channel rescalers on k, v, and FFN activations | ~0.5–1.5 M | 0.01–0.02% | ~0.02 GB | ~1% unmerged; mergeable into layer norms | `peft` |
| **BitFit** [BitFit21] | Trains only biases | ~1–3 M | 0.02–0.04% | ~0.02 GB | 0 (biases are already there) | Manual flag-flip |

## B. Prompt/activation-space methods (prepend learned vectors)

| Method | Where the delta lives | Typical trainable params | Context-length cost | Latency overhead | Notes |
|---|---|---:|---|---|---|
| **Prompt Tuning** [PromptTuning21] | Soft prompt prepended at input embedding only | 10–200 K (20 tokens × 4096 × 1) | +20 tokens | ~5% at decode (longer context) | Weak on < 10B bases |
| **Prefix Tuning** [Prefix21] | Learned key/value prefixes at every layer | 1–10 M | +20 tokens × 2 (K & V) per layer | ~10% | KV-cache enlarged permanently |
| **P-Tuning v2** [PTv2-22] | Prefix tuning + deeper per-layer prompts | 5–20 M | Same as prefix tuning | ~10% | Marketed as prompt-tuning, acts like prefix tuning |

## C. Composite methods (combinations)

| Method | Composition | Notes |
|---|---|---|
| QDoRA | NF4 base + DoRA | Same memory story as QLoRA; ~5% worse decode than QLoRA, often cited better quality on small bases |
| LoRA-FA | LoRA with A frozen after init, only B trained | Half the memory of LoRA at some quality cost |
| VeRA | LoRA with shared frozen random A, B across layers; scales learned | 10× fewer params than LoRA; quality trades off at larger bases |

## D. Quality signals (what to expect)

These are **directional** patterns from the literature on general instruction tuning
of 7B-class models; never cite them as a benchmark without rerunning.

| Scenario | Full FT | LoRA | QLoRA | DoRA | Adapters | IA³ | Prefix |
|---|---|---|---|---|---|---|---|
| General chat (MT-Bench, AlpacaEval) | 100% | 97–99% | 95–98% | 97–99% | 95–98% | 90–95% | 85–92% |
| Reasoning (GSM8K, MATH) | 100% | 93–97% | 90–95% | 94–98% | 90–95% | 80–88% | 75–85% |
| Code (HumanEval, MBPP) | 100% | 92–96% | 88–94% | 93–97% | 88–94% | 75–85% | 70–82% |
| Format-shaped (JSON, tool-use) | 100% | 98–100% | 97–99% | 98–100% | 97–100% | 90–95% | 88–94% |

LoRA and DoRA are the honest defaults in 2026 — within 3 points of full FT
on general SFT, within 5 points on hard reasoning, ~zero inference tax when
merged, and library support everywhere. IA³ / prefix tuning survive in
specific niches (very small trainable budget, per-tenant prompt
customization) but are rarely the first pick.

## E. Training compute / time (7B base, 1 epoch on 50k SFT rows, 1× A100 40GB)

Numbers are rough; they move with sequence length, batch size, and kernel.

| Method | Time to one epoch | Peak VRAM |
|---|---|---:|
| Full FT | 48 h+ (requires sharding, cf. mod-107) | > 80 GB — doesn't fit |
| LoRA (bf16 base) | ~3.5 h | ~26 GB |
| QLoRA (NF4 base) | ~4.2 h | ~15 GB |
| DoRA (bf16 base) | ~4.0 h | ~28 GB |
| Prefix tuning | ~4.8 h | ~24 GB |
| Adapters (Houlsby) | ~4.5 h | ~27 GB |
| IA³ | ~3.0 h | ~22 GB |

## F. Serving compatibility matrix

| Method | Mergeable | vLLM support | TGI support | Multi-tenant serving |
|---|---|---|---|---|
| LoRA | Yes | Yes (`--enable-lora`) | Yes | Native (routing adapters) |
| QLoRA | Yes (after dequant) | Via bitsandbytes-quantized base + LoRA | Yes | Same as LoRA |
| DoRA | Yes | Partial | Partial | Via PEFT |
| AdaLoRA | Yes | No native | No native | Via PEFT |
| Adapters | No (serial insertion) | No | No | Not really |
| IA³ | Partial (scales fold into norms) | No native | No | Via PEFT |
| Prefix / Prompt Tuning | No (prepended, not mergeable) | Partial via prompt-based APIs | Partial | Trivially tenant-keyed |
```

### 2.2 `method_decisions.md` — short decision guide

```markdown
# PEFT Method — Decision Guide

Pick along two axes: **memory you have** and **what you're adapting**.

## Axis 1 — Memory
- 24 GB consumer GPU, 7B base → **QLoRA**. No real competition.
- 40–80 GB datacenter GPU, 7B base → **LoRA** (bf16 base). Faster than QLoRA.
- 8× A100 80GB, 70B base → full FT is still rare (data+time budget); **LoRA** (sharded with FSDP, cf. mod-107).
- 80 GB VRAM, 7B base, <1M trainable params needed → **IA³** or **BitFit** (very tight budgets, mostly research).

## Axis 2 — What you're adapting
- **Behavior / style of responses** → LoRA / QLoRA / DoRA. Weight-space
  deltas are the right abstraction.
- **Prompt-shaped task, want to swap per-tenant soft prompts without model
  change** → **Prompt Tuning** or **Prefix Tuning**. Fragile at < 10B bases;
  more useful on managed-API endpoints that expose "virtual tokens."
- **Need many lightweight adapters per base, serving-efficient** → LoRA
  with vLLM `--enable-lora`. Nothing else routes cheaply.
- **Need to preserve base behavior exactly outside the adapter's domain** →
  LoRA / DoRA with low α/r and tight data mix. DoRA helps slightly here
  because magnitude/direction decomposition reduces off-target drift
  [DoRA24].

## Defaults for named scenarios
| Scenario | Default | Backup |
|---|---|---|
| Single-task SFT, 7B base, 24 GB GPU | **QLoRA** (r=16, qkvo+gate/up/down, bf16 compute) | DoRA with same targets if you can spare 10% compute |
| Multi-tenant chat, 10+ tenants | **LoRA** per tenant, served via vLLM `--enable-lora` | Prompt Tuning for small-text-style tweaks |
| Code model, 7B→13B, bf16 base | **LoRA** r=32 on attention + MLP | DoRA for reported ~2–3 pt HumanEval lift |
| Low-data instruction tuning (<5k rows) | **LoRA** r=8, dropout=0.1 | IA³ to minimize overfitting surface |
| Managed API (OpenAI-style fine-tune endpoint) | Whatever the API exposes (usually LoRA under the hood, cf. mod-110) | Prompt Tuning if the API exposes "soft prompts" |
| Research: smallest trainable budget | **IA³** or **BitFit** | LoRA r=1, α=2 as a strawman |

## When *not* to reach for PEFT
- The base is wrong for the task. Twenty LoRA adapters won't make a 1B
  model answer reasoning questions. Pick a bigger or better-aligned base
  first (mod-111 covers base-model selection).
- The task shifts the tokenizer or chat template. PEFT keeps the base's
  tokenizer; a new template means you also need `modules_to_save` for
  embeddings / lm_head, and you're approaching full FT territory.
- You need to *forget* a capability (unlearning). LoRA-style additive deltas
  can suppress but not reliably remove behavior; dedicated unlearning
  methods exist and are not covered by PEFT.
```

### 2.3 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Group methods by delta location (weight / activation / input) | The axis that predicts serving story and quality ceiling | Group by chronology or paper venue | Chronological order rewards novelty, not usefulness |
| Keep parameter counts in absolute numbers, not just % | % numbers for a 7B base don't help the reader sizing a 70B run | % only | % hides the fact that LoRA at r=16 on 7B is ~42 M params regardless of how it reads as a fraction |
| Three quality buckets (chat / reasoning / code), not a single "quality" column | The gap LoRA shows on reasoning (5 pts) is easy to miss if you average | Single score | Averages hide the variance that drives the method pick |
| Call out library support | Methods without serving support don't ship | Omit | Ignoring tooling leads to picking methods that can't move past the lab |
| Scenario defaults, not "the best method" | "Best" is empty without a scenario | Rank methods globally | Global ranking invents a context the reader doesn't have |
| Separate matrix from decision guide | Matrix is a reference, guide is advice | Fold them together | Mixing them makes both longer and harder to re-use |
| Mark soft-prompt methods as fragile under 10B | Prompt tuning is strong at 11B+ (PaLM-size), weak on 7B [PromptTuning21] | Treat them as universal | Bad default for the audience of this module, which is 7B-class fine-tuners |

## 3. Validation steps

1. **Row completeness.** Every method row in the matrix must have: trainable-param range, VRAM delta, inference overhead, mergeable yes/no, library support. Missing cells are the single most common defect; grep for empty pipes.
2. **Numbers are checkable.** For LoRA, QLoRA, DoRA: a learner should be able to reach the quoted parameter count from `{r, α, targets, n_layers, d_model, d_ff}`. Add the arithmetic in a footnote or an appendix so claims aren't invented.
3. **Links to primary sources.** Every method links to its paper (arXiv or published venue) and to the authoritative implementation (`peft`, `adapters`, or the paper's code repo). Blog-only citations fail this check.
4. **Serving story is accurate.** vLLM's `--enable-lora` supports LoRA/QLoRA; it does not support Adapters, Prefix Tuning, IA³ natively as of 2026-11-12 [vLLM-LoRA]. If the row claims otherwise, fix.
5. **Scenario advice disagrees with itself where expected.** A learner who reads the matrix and the decision guide should see the two agree. If the matrix says "DoRA costs 10% more train-time compute" and the guide says "DoRA is a drop-in LoRA replacement," one of the two is wrong.
6. **No orphan methods.** If a row exists, at least one scenario should use it or the rationale should state why it's kept as a reference only (e.g., BitFit is listed for completeness; the guide should say so).

## 4. Rubric for grading the learner's matrix

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Matrix coverage** | Missing one of {full FT, LoRA, QLoRA, Adapters, Prefix/Prompt Tuning} | All five present; missing one of {DoRA, AdaLoRA, IA³, BitFit} | All ten present with trainable-param range, VRAM, inference overhead, and mergeability for each |
| **Correctness of claims** | Any obviously wrong claim (e.g., "LoRA inserts a new MLP", "Prompt Tuning is weight-space") | Mostly right; one off-by-order-of-magnitude on param counts | All numbers defensible with an arithmetic trace or a paper citation |
| **Library + serving reality** | Claims serving support for methods that don't have it | Mentions `peft` and `adapters` but not vLLM / TGI / LoRAX | Accurately lists vLLM, TGI, LoRAX support per method |
| **Decision guide** | Single "best method" recommendation | Scenario table present but with one-line answers | Scenarios map to primary + backup method with the "why" tied to axes (memory, tenancy, quality bar) |
| **Caveats + when-not-to-use** | None | Mentions "evaluate on your task" | Explicit limitations: soft prompts under 10B, unlearning out of scope, tokenizer / template drift forcing full FT |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Treating the quality column as a benchmark. Two decimal places on "97.3%" imply a reproducible number; the matrix's job is to encode direction, not to re-score the field.
- Listing DoRA as "LoRA but better" without naming its ~10% train-time cost and the fact that unmerged inference is slower too.
- Marking Adapters as "mergeable" — they are not. Houlsby/Pfeiffer insert new sub-layers in series; there's no weight-space decomposition to fold.
- Conflating Prompt Tuning (soft prompt at input) with P-Tuning v2 (prefixes at every layer). They share a lineage but differ in where the delta lives.
- Quoting training throughput without hardware. "~3.5 h per epoch" without "on 1× A100 40GB at seq 1024 bs 16" is noise.
- Recommending IA³ as a universal cheap-training option without naming the 5–10 pt quality gap on reasoning and code at the 7B scale.
- Omitting the "when not to PEFT" row — the exercise grades for completeness and intellectual honesty, and the honest answer is sometimes "pick a bigger base."
- Citing one blog post per method instead of the paper and the authoritative implementation. The matrix is a reference; references should be durable.
- Treating "mergeable" as a boolean that implies "and therefore served at zero cost." Serving LoRA merged requires one checkpoint per task; unmerged routes but costs 3% latency. Say both.
- Leaving full fine-tuning out of the table. It's the baseline every PEFT method gets scored against; cutting it makes the trade-offs invisible.

## 6. References

- [LoRA21] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021. <https://arxiv.org/abs/2106.09685>
- [QLoRA23] Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," 2023. <https://arxiv.org/abs/2305.14314>
- [DoRA24] Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation," 2024. <https://arxiv.org/abs/2402.09353>
- [AdaLoRA23] Zhang et al., "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning," ICLR 2023. <https://arxiv.org/abs/2303.10512>
- [Houlsby19] Houlsby et al., "Parameter-Efficient Transfer Learning for NLP," ICML 2019. <https://arxiv.org/abs/1902.00751>
- [Pfeiffer20] Pfeiffer et al., "AdapterFusion: Non-Destructive Task Composition for Transfer Learning," 2020. <https://arxiv.org/abs/2005.00247>
- [IA3-22] Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning," NeurIPS 2022. <https://arxiv.org/abs/2205.05638>
- [BitFit21] Zaken, Ravfogel, Goldberg, "BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models," 2021. <https://arxiv.org/abs/2106.10199>
- [Prefix21] Li, Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation," 2021. <https://arxiv.org/abs/2101.00190>
- [PromptTuning21] Lester, Al-Rfou, Constant, "The Power of Scale for Parameter-Efficient Prompt Tuning," 2021. <https://arxiv.org/abs/2104.08691>
- [PTv2-22] Liu et al., "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks," 2022. <https://arxiv.org/abs/2110.07602>
- [vLLM-LoRA] vLLM multi-LoRA serving docs. <https://docs.vllm.ai/en/latest/features/lora.html>
- Hugging Face `peft` library. <https://github.com/huggingface/peft>
- AdapterHub / `adapters` library. <https://github.com/adapter-hub/adapters>
- OWASP Machine Learning Security Top 10 (ML-06 AI Supply Chain Vulnerabilities — adapter method choice affects provenance obligations). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — ML supply chain and model-artifact tactics. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Govern/Map — documenting method choice is a model-card obligation). <https://www.nist.gov/itl/ai-risk-management-framework>
