# mod-101 · exercise-01 — Decoder-Only Forward Pass Walkthrough — Solution

## 1. Solution overview

A decoder-only language model (GPT-family, Llama-family, Mistral, Qwen, etc.) turns a token-id sequence into a probability distribution over the next token by:

1. Looking up token embeddings.
2. Running the sequence through `N` identical transformer blocks, each of which applies normalization, causal self-attention, and a position-wise MLP with two residual connections.
3. Applying a final normalization and projecting to vocabulary logits via an LM head (often weight-tied to the input embedding).

For fine-tuners the forward pass matters because every design decision downstream — KV-cache layout, label masking, LoRA target modules, quantization kernels, packing strategy — is a statement about which tensors in this pass exist and what shape they have. This solution walks the pass end to end for a Llama-style architecture, annotates shapes, and lists where common fine-tuning bugs attach.

Architectural reference target (concrete so shapes are checkable):

| Symbol | Value | Meaning |
|---|---|---|
| `B` | 2 | batch size |
| `T` | 8 | sequence length (tokens) |
| `V` | 128256 | vocab size (Llama 3 tokenizer) |
| `d_model` | 4096 | hidden size |
| `n_layers` | 32 | transformer blocks |
| `n_heads` | 32 | query heads |
| `n_kv_heads` | 8 | KV heads (grouped-query attention) |
| `d_head` | 128 | per-head dim (`d_model / n_heads`) |
| `d_ff` | 14336 | SwiGLU intermediate size |
| `rope_base` | 500000 | RoPE θ base (Llama 3.x) |

Numbers above are the published Llama 3 8B config [L3]. The walkthrough logic is architecture-agnostic for any decoder-only transformer with pre-norm, RMSNorm, RoPE, GQA, and SwiGLU. If your reference base is different, swap the numbers — the shape algebra is unchanged.

## 2. Implementation — worked answer

### 2.1 Input → token ids

The tokenizer emits `input_ids: int64[B, T]`. That tensor is the only thing the model proper sees — **every** semantic choice about where turns begin, where the system prompt ends, and which tokens the loss attends to is already baked in before this step. (This is exercise-02's territory; the forward pass takes it on faith.)

### 2.2 Embedding lookup

```
tok_emb = embed_tokens[input_ids]        # float[B, T, d_model]
```

`embed_tokens` is a `V × d_model` parameter. The lookup is a gather, not a matmul. It is also the single largest parameter in small/mid-size decoder-only models (`V × d_model` ≈ 525M params at Llama 3 8B's config), which is why it is almost always tied to the LM head.

Notable: there is **no additive positional embedding here** in modern decoder-only transformers. Positional information is applied inside attention via RoPE (see §2.4). The original GPT-2/BERT-style learned positional embedding added at this step is a legacy design [AIAYN, L3].

### 2.3 Pre-attention RMSNorm

Pre-norm + RMSNorm is the dominant choice [Z19, N19]:

```
x_norm = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps) * gamma
#         [B, T, d_model]                                     [d_model]
```

Differences from LayerNorm that matter for the walkthrough:

- No mean subtraction and no bias term — strictly a rescale by RMS.
- One parameter vector `gamma` per norm (half the state of LayerNorm).
- Numerically more forgiving at low precision; `eps` lives inside the `rsqrt` to avoid cancellation.

Pre-norm (normalize *before* the sublayer, then add the residual back) is what makes deep decoder-only models trainable without warm-up-only stability hacks [X20].

### 2.4 Causal self-attention (with GQA + RoPE)

Shapes inside one layer, with GQA:

```
q = x_norm @ W_q      # [B, T, n_heads * d_head]      → view [B, T, n_heads, d_head]
k = x_norm @ W_k      # [B, T, n_kv_heads * d_head]   → view [B, T, n_kv_heads, d_head]
v = x_norm @ W_v      # same shape as k
```

With `n_heads=32`, `n_kv_heads=8`, each K/V head is shared across 4 Q heads. That is the whole point of GQA [GQA23]: the per-token KV-cache state goes from `2 * n_heads * d_head = 8192` floats down to `2 * n_kv_heads * d_head = 2048` floats — a 4× memory cut at long context without the quality collapse of full MQA.

**RoPE application.** Rotary position embedding rotates `q` and `k` in each 2-dim channel pair by an angle `m * θ_i`, where `m` is the absolute position and `θ_i = base^(-2i/d_head)` [RoPE22]:

```
q_rot[..., 2i]   =  q[..., 2i]   * cos(m*θ_i) - q[..., 2i+1] * sin(m*θ_i)
q_rot[..., 2i+1] =  q[..., 2i]   * sin(m*θ_i) + q[..., 2i+1] * cos(m*θ_i)
# same for k
```

V is **not** rotated. RoPE's property is that `⟨q_rot_m, k_rot_n⟩` depends only on the relative offset `m - n`, which is what lets attention generalize across positions. Context-length extension (position interpolation / NTK / YaRN) rescales `θ` or `m` — see exercise-03.

**Scaled dot-product attention with causal mask.** Expand KV heads to match Q heads (GQA-expand is a cheap `repeat_interleave`), then:

```
scores = (q_rot @ k_rot.transpose(-2,-1)) / sqrt(d_head)    # [B, n_heads, T, T]
scores = scores + causal_mask                               # upper triangle = -inf
attn   = softmax(scores, dim=-1)
out    = attn @ v                                           # [B, n_heads, T, d_head]
out    = out.reshape(B, T, d_model) @ W_o                   # [B, T, d_model]
```

The causal mask is additive `-inf` on the upper triangle so that position `t` cannot see positions `> t`. This is what turns the transformer into a language model; removing it is bidirectional (encoder) attention.

**Residual:**

```
x = x + attn_out
```

### 2.5 Pre-MLP RMSNorm + SwiGLU MLP

```
x_norm2 = rmsnorm(x, gamma_mlp)                             # [B, T, d_model]
gate    = x_norm2 @ W_gate                                  # [B, T, d_ff]
up      = x_norm2 @ W_up                                    # [B, T, d_ff]
ff      = silu(gate) * up                                   # elementwise gated activation
x       = x + ff @ W_down                                   # [B, T, d_model], residual
```

SwiGLU is the dominant modern choice over ReLU/GELU [S20]. `d_ff` is typically `~2.7 × d_model` rather than the classical `4 × d_model`, because SwiGLU uses three weight matrices instead of two and the budget was rebalanced.

### 2.6 Stack, final norm, LM head

Loop §2.3–2.5 for `n_layers` blocks, then:

```
h_final = rmsnorm(x, gamma_final)                           # [B, T, d_model]
logits  = h_final @ W_lm_head.T                             # [B, T, V]
```

`W_lm_head` is usually the tied `embed_tokens`. Logits have the full vocab on the last axis; sampling, cross-entropy, and any custom head (reward, classifier) tap in here.

### 2.7 Loss (training only)

Standard next-token CE with a shift:

```
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
loss = cross_entropy(shift_logits.view(-1, V),
                     shift_labels.view(-1),
                     ignore_index=-100)
```

`ignore_index=-100` is how HuggingFace masks prompt tokens, pad tokens, and anything else the fine-tuner wants excluded from the loss. Getting this mask right is where most SFT bugs actually live.

### 2.8 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Pre-norm vs post-norm | Trainable at depth without warm-up-only stability [X20] | Post-norm (original Transformer) | Post-norm can be slightly higher quality at small scale but needs careful LR schedule |
| RMSNorm vs LayerNorm | Fewer ops, no mean-subtract, more stable in low precision [Z19] | LayerNorm | LayerNorm is marginally more expressive; parity of final quality is empirical |
| SwiGLU vs GELU | Better validation loss at matched params [S20] | GELU / ReLU | 1.5× more weight matrices in the MLP block |
| GQA vs MHA vs MQA | KV-cache cut at long ctx with near-MHA quality [GQA23] | Full MHA (KV-heavy) / MQA (quality loss) | Picks a point on memory ↔ quality curve |
| RoPE vs learned/absolute | Relative encoding, extends to longer contexts via θ scaling [RoPE22] | ALiBi [ALiBi21], learned absolute | See exercise-03 |
| Weight-tied LM head | Saves `V×d_model` params; proven no-quality-loss | Untied | Untied gives a tiny bit of flexibility for separate final-layer LR |

### 2.9 Where fine-tuning bugs attach to this pass

- **Label mask misalignment** (§2.7): prompt tokens counted in the loss; or the shift by one dropped; or the EOS ignored.
- **KV-cache position drift** (§2.4): resuming generation without tracking absolute position breaks RoPE rotations.
- **Dtype mismatches around norm** (§2.3, §2.5): norm cast to fp32 but gamma left in bf16 mixes dtypes silently.
- **LoRA target picks** (§2.4, §2.5): adapting only `W_q, W_v` is the LoRA paper's recipe [LoRA21]; adapting `W_o, W_gate, W_up, W_down` matters for larger quality gains and is what most modern recipes do.
- **Flash-attention causal flag** (§2.4): forgetting `is_causal=True` turns a decoder into a bidirectional model that still trains but leaks future tokens.

## 3. Validation steps

A learner can self-check with:

1. **Shape sanity.** Pick a tiny Llama/Qwen checkpoint (e.g. a 0.5B–1B model). Load with `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32)`. Register a forward hook on every submodule and record input/output shapes for one `B=2, T=8` batch. Each block should show the shapes table in §2.2–§2.6.
2. **Residual accounting.** Confirm that removing either residual (patch `block.forward` to skip the add) collapses the loss — a fast end-to-end check that both residuals are load-bearing.
3. **Causal mask check.** Replace `is_causal=True` with `False` in a copy; next-token loss on held-out text should *drop below trained baseline* (future-token leak). This is the quickest proof the causal mask is actually masking.
4. **RoPE rotation check.** Dump `q_rot` and `k_rot` for the same token at positions `m=0` and `m=10`. The vectors differ; their inner product with a fixed reference token's `k` depends only on the offset. (Su et al. [RoPE22] §3.4.)
5. **Weight-tie check.** `model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()` for tied checkpoints.
6. **Logits-to-loss round-trip.** Compute `loss` two ways — via `model(..., labels=labels).loss` and manually via `cross_entropy(shift_logits, shift_labels)`. They should agree to ~1e-6 in fp32.

## 4. Rubric

Grade on the five dimensions; 2 points each; 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Shape fidelity** | Shapes wrong or missing | Shapes present but some wrong in attention/MLP | Every stage has correct shapes including GQA KV expansion |
| **Positional encoding** | RoPE confused with absolute/learned | RoPE described but no rotation formula or V-rotation error | Correct per-channel-pair rotation; V unrotated; relative-position property named |
| **Causal masking + loss** | Causal mask missing; no shift | One of {causal, shift, ignore_index} wrong | All three correct and tied to §2.7 code |
| **Architecture rationale** | Vague "it's standard" | Names the choices (pre-norm, RMSNorm, SwiGLU, GQA) without trade-off | Choice + alternative + the trade-off, grounded in a cited paper |
| **Validation** | None or hand-wavy | One concrete check | At least three concrete, runnable checks from §3 |

**Pass threshold:** 7/10, with no single dimension at 0.

## 5. Common mistakes

- Treating RoPE as an *additive* positional embedding at the token-embedding step. It is a *rotation* applied inside attention, to `q` and `k` only.
- Forgetting the shift when going from logits to loss. `labels[t]` is the target at position `t`; the model's logit for that target lives at `logits[t-1]`.
- Omitting the GQA head expansion in the Q·Kᵀ step and getting a shape error that gets "fixed" with a wrong reshape.
- Counting `embed_tokens` and `lm_head` as two parameter tensors when they are tied — doubles the reported parameter count.
- Using `n_heads` where `n_kv_heads` belongs when sizing the KV cache (`2 × B × n_kv_heads × T × d_head × dtype_bytes`).
- Applying the causal mask after softmax instead of before (makes attention leak).
- Describing attention "scores" as probabilities before softmax.

## 6. References

- [AIAYN] Vaswani et al., "Attention Is All You Need," NeurIPS 2017. <https://arxiv.org/abs/1706.03762>
- [X20] Xiong et al., "On Layer Normalization in the Transformer Architecture," ICML 2020. <https://arxiv.org/abs/2002.04745>
- [Z19] Zhang & Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019. <https://arxiv.org/abs/1910.07467>
- [N19] Nguyen & Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention," IWSLT 2019. <https://arxiv.org/abs/1910.05895>
- [S20] Shazeer, "GLU Variants Improve Transformer," 2020. <https://arxiv.org/abs/2002.05202>
- [GQA23] Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," EMNLP 2023. <https://arxiv.org/abs/2305.13245>
- [RoPE22] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2022. <https://arxiv.org/abs/2104.09864>
- [ALiBi21] Press, Smith, Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation," ICLR 2022. <https://arxiv.org/abs/2108.12409>
- [LoRA21] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021. <https://arxiv.org/abs/2106.09685>
- [L3] Meta, "The Llama 3 Herd of Models," 2024. <https://arxiv.org/abs/2407.21783>
- HuggingFace Transformers, `modeling_llama.py` source. <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py>
