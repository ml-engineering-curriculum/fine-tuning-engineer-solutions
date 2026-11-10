# mod-102 · exercise-03 — Chat Template and Multi-Turn Formatting — Solution

## 1. Solution overview

The exercise forces one invariant to the surface: *the tokens the loss sees are what the model learns to emit.* Everything else — the chat prefix, the system prompt, the user turns — is context. Getting the mask wrong is the single most expensive SFT bug because it does not crash: the loss just goes down for the wrong reason.

The solution renders one authored multi-turn conversation through two template families — Llama 3 (`<|start_header_id|>...<|end_header_id|>`, terminated by `<|eot_id|>`) [L3] and ChatML (`<|im_start|>role\n...\n<|im_end|>`) [QWEN] — produces trainer-ready `input_ids / attention_mask / labels`, and verifies five invariants that catch the drift bugs real pipelines hit.

The core trick for labels: use `apply_chat_template(..., return_assistant_tokens_mask=True)` when the installed `transformers` version supports it [HFTPL]. When it does not, rebuild labels from the full-conversation token sequence by finding the token spans between each assistant-turn start marker and its terminator — never by re-tokenizing the response string on its own.

## 2. Worked implementation

### 2.1 `conversation.json` (Part A)

```json
[
  {"role": "system",
   "content": "You are a senior Python engineer. Prefer stdlib. Explain tersely. When code is asked for, respond with a single fenced code block and one sentence above it describing what the snippet does. Never include speculative benchmarks."},
  {"role": "user",
   "content": "I have this function and it's slow. Can you make it faster without changing behavior?\n```python\ndef unique_sorted(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return sorted(out)\n```"},
  {"role": "assistant",
   "content": "Replace the O(n^2) membership check with a set.\n```python\ndef unique_sorted(xs):\n    return sorted(set(xs))\n```"},
  {"role": "user",
   "content": "What if the elements are unhashable (e.g. dicts)? Rewrite for that case."},
  {"role": "assistant",
   "content": "Fall back to a seen-list using equality comparison; stable order is retained.\n```python\ndef unique_sorted_unhashable(xs, key=repr):\n    seen = set(); out = []\n    for x in xs:\n        k = key(x)\n        if k not in seen:\n            seen.add(k); out.append(x)\n    return sorted(out, key=key)\n```"}
]
```

### 2.2 `build_sft_example` (Part C)

```python
# build_sft_example.py
from __future__ import annotations
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

IGNORE_INDEX = -100

def build_sft_example(tokenizer: PreTrainedTokenizerBase, messages: list[dict]) -> dict:
    """Return input_ids / attention_mask / labels for an SFT training example.

    Masks every non-assistant token to -100. Keeps the assistant turn
    terminator (<|eot_id|> / <|im_end|>) in the loss so the model learns
    to stop.
    """
    # Preferred path: let the template engine emit the mask directly.
    try:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = enc["input_ids"]
        assistant_mask = enc["assistant_masks"]
    except (TypeError, ValueError):
        # Fallback: compute the mask ourselves (see §2.3).
        input_ids, assistant_mask = _fallback_mask(tokenizer, messages)

    labels = [tid if m == 1 else IGNORE_INDEX
              for tid, m in zip(input_ids, assistant_mask)]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels":         torch.tensor(labels, dtype=torch.long),
    }
```

### 2.3 Fallback mask when `return_assistant_tokens_mask` is unavailable

The robust fallback renders *prefix tokens* for each assistant turn and matches them as sub-sequences of the full render. This is the only approach that correctly keeps the terminator (`<|eot_id|>` / `<|im_end|>`) inside the loss while keeping system/user/tool tokens out.

```python
def _fallback_mask(tokenizer, messages):
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    mask = [0] * len(full)

    # For each assistant turn, compute: prefix (everything up to and including
    # the assistant header marker) and post-content tokens (through terminator).
    # Mark [prefix_end, terminator_end] as 1.
    cursor = 0
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prior = messages[:i]
        prior_with_header = prior + [{"role": "assistant", "content": ""}]
        # Length of the prior conversation + just the "assistant\n" header.
        header_len = len(tokenizer.apply_chat_template(
            prior_with_header, tokenize=True, add_generation_prompt=False,
        )) - _trailing_empty_assistant_tokens(tokenizer)
        this_through = prior + [messages[i]]
        end_len = len(tokenizer.apply_chat_template(
            this_through, tokenize=True, add_generation_prompt=False,
        ))
        # Mark the response tokens and the terminator.
        for t in range(header_len, end_len):
            mask[t] = 1
        cursor = end_len
    return full, mask

def _trailing_empty_assistant_tokens(tokenizer) -> int:
    """How many tokens does an empty assistant content contribute in this template?
    (i.e., the terminator alone)
    """
    a = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": ""}],
        tokenize=True, add_generation_prompt=False,
    )
    b = tokenizer.apply_chat_template(
        [], tokenize=True, add_generation_prompt=False,
    )
    return len(a) - len(b)
```

(The `transformers` 4.44+ built-in path is strictly preferred. The fallback exists so the exercise does not depend on a new version of a library the learner may not control.)

### 2.4 Rendering + ID-table printer (Part B)

```python
# render_and_print.py
from transformers import AutoTokenizer

MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

def dump(tok, messages):
    train = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=False)
    infer = tok.apply_chat_template(messages[:-1], tokenize=False,
                                    add_generation_prompt=True)
    ids = tok.apply_chat_template(messages, tokenize=True,
                                  add_generation_prompt=False)
    print(f"--- {tok.name_or_path} ---")
    print("[train render]")
    print(repr(train))
    print("[infer render for last-prompt]")
    print(repr(infer))
    print("[id table]")
    for i, tid in enumerate(ids):
        print(f"{i:>4}  {tid:>6}  {tok.decode([tid])!r}")
```

### 2.5 Expected patterns (worked observation)

**Llama 3.1 training render** starts with `<|begin_of_text|>` and uses role headers `<|start_header_id|>system<|end_header_id|>\n\n…<|eot_id|>` [L3]. Each assistant turn ends with `<|eot_id|>` (token 128009), not `<|end_of_text|>` (128001).

**Qwen 2.5 training render** starts with no BOS (ChatML convention) and uses `<|im_start|>role\n…\n<|im_end|>` [QWEN]. The terminator is `<|im_end|>`.

Surprises worth noting in the write-up:

- Llama 3's chat template inserts a date in the default system block when `add_generation_prompt=True` — if the learner does not pass an explicit system message, a date leak enters the render. Check your tokenizer's chat template string.
- Qwen's template does not insert BOS; concatenating two renders will double-start-sequence only if you intentionally add one.
- Both templates place a newline between header and content; the mask must include that newline *only* for assistant turns.

### 2.6 Invariant assertions (Part D)

```python
# invariants.py
def assert_invariants(tokenizer, messages, example):
    input_ids = example["input_ids"].tolist()
    labels    = example["labels"].tolist()

    # (1) Prompt prefix match.
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
    )
    assert input_ids[: len(prompt_ids)] == prompt_ids, (
        "add_generation_prompt=True render is not a prefix of the full render; "
        "chat template is non-deterministic between the two modes."
    )

    # (2) Final token is a known turn terminator.
    terminators = set()
    for tok in ("<|eot_id|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            terminators.add(tid)
    assert input_ids[-1] in terminators, (
        f"final token {input_ids[-1]} is not a known terminator {terminators}"
    )

    # (3) All non-assistant tokens masked.
    # Rebuild a role-per-token tag from the fallback mask, then check.
    _, assistant_mask = _fallback_mask(tokenizer, messages)
    for i, (lab, in_a) in enumerate(zip(labels, assistant_mask)):
        if in_a == 0:
            assert lab == -100, f"pos {i} is non-assistant but label is {lab}"

    # (4) All assistant tokens kept.
    for i, (tid, lab, in_a) in enumerate(zip(input_ids, labels, assistant_mask)):
        if in_a == 1:
            assert lab == tid, f"pos {i} is assistant but label={lab} ≠ input={tid}"

    # (5) Active-token ratio in a sane band.
    active = sum(1 for l in labels if l != -100)
    ratio = active / len(labels)
    assert 0.15 <= ratio <= 0.85, f"active ratio {ratio:.2f} out of [0.15, 0.85]"
```

### 2.7 Bug introduction + catch (Part E) — worked example

**Chosen bug:** *Rebuild `labels` from the string offset of `messages[-1]["content"]` in the rendered text.*

Broken code:

```python
def broken_build(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False)
    last = messages[-1]["content"]
    start_char = rendered.rfind(last)
    end_char   = start_char + len(last)
    ids = tokenizer(rendered, return_offsets_mapping=True,
                    add_special_tokens=False)
    labels = []
    for tid, (a, b) in zip(ids["input_ids"], ids["offset_mapping"]):
        inside = a >= start_char and b <= end_char
        labels.append(tid if inside else -100)
    return {"input_ids": torch.tensor(ids["input_ids"]),
            "attention_mask": torch.tensor([1]*len(ids["input_ids"])),
            "labels": torch.tensor(labels)}
```

**What the invariants catch.** On the authored conversation, the first assistant turn is still masked; only the final assistant turn is trained on. Invariant (4) fires at the first assistant turn's first content token:

```
AssertionError: pos 52 is assistant but label=-100 ≠ input=128006
```

**Fix.** Use the full-render + sub-sequence mask approach from §2.3 (or the built-in `return_assistant_tokens_mask`). The mask runs across every assistant turn and includes each turn's terminator, so earlier assistant turns are not silently dropped from the loss.

**Follow-ups a reviewer should ask for.** (i) A regression test on a fuzzed set of 2–5 turn conversations. (ii) A runtime log on training showing the active-token ratio per batch — if it suddenly halves after a template update, the bug has returned.

### 2.8 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| `return_assistant_tokens_mask=True` when available | Native, correct, tokenizer-agnostic | Hand-rolled mask | Hand-rolled drifts when templates change (hit by the bug in §2.7) |
| Full-render + sub-sequence fallback | Guarantees `prompt_render + response_tokens` identity | Re-tokenize response separately | Re-tokenizing adds/removes BOS/whitespace, breaks invariant 1 |
| Keep the terminator in the loss | Model has to learn when to stop; omitting it trains models that over-generate | Mask the terminator | Terminator-masked models trail off |
| Train on *every* assistant turn | Multi-turn behavior is the training target | Train only on the last turn | Last-turn-only wastes 1–3 assistant turns of supervision per example |
| Both renders diffed visually | Catches silent injection (date header, default system prompt) | Trust the template | Silent additions escape into data |
| Active-ratio bound 0.15–0.85 | Catches both "everything masked" and "nothing masked" at a glance | No ratio bound | The two most common bugs (all-mask, no-mask) sail through |

## 3. Validation steps

1. **Round-trip decode.** `tokenizer.decode(input_ids)` must reconstruct the render string character-for-character (modulo the leading BOS for Llama).
2. **Prompt-prefix invariant.** `prompt_render_ids == full_render_ids[:len(prompt_render_ids)]`. If false, print both token sequences and find the divergence; the chat template is non-idempotent between the two modes and will silently poison training.
3. **Terminator check.** `input_ids[-1]` is `<|eot_id|>` (Llama 3 → 128009) or `<|im_end|>` (Qwen 2.5). Any other final token is wrong.
4. **Per-position mask read.** Dump the `(position, decoded_token, label_is_active)` table and read it end-to-end once. Confirm *every* user/system/tool span is "mask" and *every* assistant span (through terminator) is "train".
5. **Active ratio sanity.** `sum(l != -100 for l in labels) / len(labels)` in `[0.15, 0.85]`. The authored conversation sits around 0.35.
6. **Fuzz regression.** Build 100 random 2–6-turn conversations with short random user/system/assistant content; run all five invariants against both tokenizers; any failure is a template-interaction bug to fix before it reaches training.

## 4. Rubric

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Rendering** | One tokenizer only, or renders missing visible whitespace/special tokens | Both renders, but no visible diff between training and inference | Both tokenizers, both modes, token-ID table dumped, surprises noted |
| **Trainer tensors** | Labels incorrectly derived or terminator excluded | `build_sft_example` exists but only trains on final assistant turn | Labels mask all non-assistant tokens, keep terminator, train every assistant turn |
| **Invariant coverage** | <3 of 5 invariants, or all pass trivially | All 5 defined, but one hand-waves (e.g. role tag not recomputed) | All 5 defined and passing on both tokenizers with a real conversation |
| **Bug round-trip** | No bug introduced, or bug introduced but no invariant catches it | Bug introduced; invariant catches it but no fix shown | Bug + failing invariant + fix all in the write-up; ≤2 paragraphs |
| **Visualization + read** | No mask visualization | Visualization exists but was not read span-by-span | Visualization exists and the write-up names the spans trained/masked |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Re-tokenizing the assistant response separately. BOS/EOS/whitespace injection shifts token boundaries; the resulting `labels` are off by one or more positions per turn. Always derive labels from the full render.
- Masking the terminator (`<|eot_id|>` / `<|im_end|>`). The model then never learns to stop and ends every response mid-thought at inference.
- Training only on the last assistant turn. Every earlier assistant turn is wasted supervision. On a 4-turn conversation, this is a 2× data loss.
- Setting `pad_token = eos_token` and then including pad positions in the loss. Use a distinct pad token or explicitly mask pads to `-100`.
- Using `add_generation_prompt=True` for training. That render ends with the assistant *header* but no content; it is the inference render.
- Accepting the `add_generation_prompt=True` render as the "prompt" without checking it is a strict token-prefix of the training render. Some templates are not deterministic between the two modes (date headers are a common offender).
- Forgetting that Llama's `<|end_of_text|>` (128001) is *not* the turn terminator; `<|eot_id|>` (128009) is.
- Reading the loss-mask visualization only around the final turn. The bug lives in the middle turns.

## 6. References

- [L3] Meta, "The Llama 3 Herd of Models," 2024. Chat template + special tokens appendix. <https://arxiv.org/abs/2407.21783>
- [QWEN] Qwen Team, "Qwen2.5 Technical Report," 2024. ChatML format. <https://arxiv.org/abs/2412.15115>
- [HFTPL] HuggingFace Transformers chat-templates documentation (`return_assistant_tokens_mask`). <https://huggingface.co/docs/transformers/main/en/chat_templating>
- [TRL] HuggingFace TRL — `DataCollatorForCompletionOnlyLM` as a reference implementation of assistant-only masking. <https://huggingface.co/docs/trl/main/en/sft_trainer>
- Axolotl — `train_on_inputs: false` masking recipe. <https://github.com/OpenAccess-AI-Collective/axolotl>
- OWASP Machine Learning Security Top 10 — ML-02 Data Poisoning context (mask-mismatch is an under-the-hood label flip). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — training-data integrity. <https://atlas.mitre.org/>
- NIST AI RMF — Measure/Manage. <https://www.nist.gov/itl/ai-risk-management-framework>
