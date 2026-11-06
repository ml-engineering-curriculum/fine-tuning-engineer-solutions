# mod-101 · exercise-02 — Tokenizer and Chat Template Audit — Solution

## 1. Solution overview

A tokenizer-and-chat-template audit is the "contract review" of a fine-tune. Before touching training config or data mixes, the fine-tuner answers four questions about the base model:

1. **How are strings turned into ids?** (algorithm, vocab size, byte-fallback, normalizer, pre-tokenizer)
2. **Which ids are special?** (BOS, EOS, PAD, UNK, role/turn markers, tool tokens, reserved slots)
3. **What is the chat template?** (the Jinja string that turns `[{"role","content"}, …]` into text, where role markers land, where generation starts)
4. **Which tokens must the loss ignore?** (the label mask policy derived from the template)

Every post-training bug that looks like "model keeps talking after it should stop", "model responds in the wrong role", "loss decreases but eval degrades after ~1 epoch", "assistant replies with the system prompt text" traces back to one of these four being wrong. This solution audits a Llama-3-style instruct model as the worked example; the audit *procedure* works for any base. HuggingFace Transformers is the reference tooling [HF-Tok, HF-CT].

## 2. Implementation — worked audit

Audit target (worked example): a Llama 3.1 Instruct base. Any `*-Instruct` / `*-Chat` checkpoint works for the procedure.

### 2.1 Pull the tokenizer and inventory it

```python
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # or any instruct base
tok = AutoTokenizer.from_pretrained(MODEL_ID)

print("Class          :", type(tok).__name__)
print("Vocab size     :", tok.vocab_size)
print("Model max len  :", tok.model_max_length)
print("Fast tokenizer :", tok.is_fast)
print("BOS / EOS / PAD:", tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
print("UNK            :", tok.unk_token_id)
print("Padding side   :", tok.padding_side)
print("Truncation side:", tok.truncation_side)
```

Record these in the audit report verbatim. Four fields drive a lot of downstream behavior:

- **`pad_token_id`.** Many Llama-family checkpoints ship with `pad_token_id=None`. If left `None`, `DataCollatorForLanguageModeling` and friends will either crash or silently reuse `eos_token_id` for padding — and then the EOS loss contribution is diluted across all the pad slots. Set `pad_token_id` explicitly (often to a reserved special token such as `<|finetune_right_pad_id|>` for Llama 3.1) [L3].
- **`padding_side`.** Training: left or right is fine so long as the attention mask is correct. Decoding: **left-pad** so that the last real token sits at the same column across the batch — otherwise generation starts from the padded column on short rows.
- **`model_max_length`.** The *tokenizer's* idea of max length. Sanity-check it matches the model config; HF ships mismatches occasionally.
- **`is_fast`.** Use the Rust-backed `PreTrainedTokenizerFast`. The slow Python path is fine for correctness but will bottleneck data preprocessing at any real scale.

### 2.2 Inventory the special tokens

```python
import json
print(json.dumps(tok.special_tokens_map, indent=2))
print("Added tokens:")
for tid, t in sorted(tok.added_tokens_decoder.items()):
    print(f"  {tid:>6}  {t.content!r:>40}  "
          f"special={t.special}  single_word={t.single_word}")
```

For Llama 3, you will see at minimum `<|begin_of_text|>`, `<|end_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>` (end-of-turn), plus reserved slots `<|reserved_special_token_*|>` [L3]. Flag three things for the audit:

- **EOS vs EOT.** Instruct models typically stop on `<|eot_id|>` (end-of-turn) per turn, with `<|end_of_text|>` reserved for end-of-document. Fine-tuners who hard-code `eos_token_id=end_of_text` get models that run past the end of a turn.
- **Reserved slots.** If the fine-tune adds a *new* role (e.g. a `tool_result` role), repurpose a reserved slot — don't invent a brand-new token id, because that forces resizing the embedding matrix and re-initializing untrained rows.
- **`single_word=True`.** This forces the tokenizer to treat the token as its own word boundary (no merging with neighbors), which is what you want for role markers.

### 2.3 Dump and read the chat template

```python
print(tok.chat_template)   # Jinja string

msgs = [
    {"role": "system",    "content": "You are a terse assistant."},
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "4."},
    {"role": "user",      "content": "And 3+3?"},
]
text = tok.apply_chat_template(msgs, tokenize=False,
                               add_generation_prompt=True)
print(text)
```

For Llama 3, that renders (abbreviated) as:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a terse assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

4.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

And 3+3?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```

Read off:
- **BOS placement.** Exactly one `<|begin_of_text|>` at the start.
- **Role framing.** Each turn is `<|start_header_id|>ROLE<|end_header_id|>\n\nCONTENT<|eot_id|>`.
- **`add_generation_prompt=True`** emits the final `assistant` header with no content, so the model is cued to generate.
- **Trailing newline.** Present or absent; template-dependent. Match the base model's convention exactly — a single trailing-newline difference between training and inference is a frequent silent regression.

### 2.4 Produce labels (the mask)

The forward-pass contract (exercise-01 §2.7) is: `ignore_index=-100` on every token the loss must skip. For a chat SFT dataset, the loss should land on **assistant tokens only** (and typically on the assistant's `<|eot_id|>` so the model learns where to stop).

The correct way to do this is to mask with the template's structure, not with regex on raw text. HuggingFace's `apply_chat_template` has a `return_assistant_tokens_mask=True` option when the template declares `{% generation %}` blocks [HF-CT]:

```python
batch = tok.apply_chat_template(
    msgs,
    return_tensors="pt",
    return_dict=True,
    return_assistant_tokens_mask=True,   # requires {% generation %} in template
)
input_ids    = batch["input_ids"]
assist_mask  = batch["assistant_masks"]       # 1 on assistant tokens, 0 elsewhere

labels = input_ids.clone()
labels[assist_mask == 0] = -100
```

If the template does not include `{% generation %}` blocks, add them (one line edit in the Jinja) or fall back to the structural approach: tokenize each turn separately with the template and record offsets, then set the non-assistant spans to `-100`. The `apply_chat_template` + `return_assistant_tokens_mask` path is the method HuggingFace documents and the one to prefer [HF-CT].

Validate the mask on one example by printing `(token, label)` pairs:

```python
for tid, lbl in zip(input_ids[0].tolist(), labels[0].tolist()):
    marker = "  LOSS" if lbl != -100 else ""
    print(f"{tid:>7}  {tok.decode([tid])!r:>30}{marker}")
```

Only assistant-content tokens (and ideally the assistant's closing `<|eot_id|>`) should carry `LOSS`.

### 2.5 Round-trip and encoding-boundary checks

```python
# Round-trip
original = "Hello, world! 🌍 — naïve café."
ids      = tok.encode(original, add_special_tokens=False)
back     = tok.decode(ids)
assert back == original, (original, back)

# Byte-fallback for unknown chars (Llama 3 uses BPE over bytes — no true UNK)
assert tok.unk_token_id is None or tok.unk_token_id not in ids

# Whitespace sensitivity
print(tok.encode("Hello"))
print(tok.encode(" Hello"))   # usually a different id — this matters for prompt joins
```

Three things to record in the audit:
- Does the tokenizer preserve the exact input string on round-trip (byte-level BPE) or does it normalize (NFC, lowercasing, extra whitespace collapse)?
- How does it handle leading whitespace? SentencePiece-derived tokenizers prepend `▁` (Llama 2) or treat leading space as part of the next token (Llama 3).
- Is there a true UNK, or byte-fallback that guarantees no OOV?

### 2.6 Prompt-side packing sanity check

For a two-turn example, verify that tokenizing the whole conversation at once produces the *same* ids as tokenizing each turn with the template and concatenating:

```python
all_at_once = tok.apply_chat_template(msgs, tokenize=True,
                                      add_generation_prompt=False)
piecewise   = []
for i, _ in enumerate(msgs):
    piecewise += tok.apply_chat_template(
        msgs[:i+1], tokenize=True, add_generation_prompt=False
    )[len(piecewise):]
# Compare first N tokens — a divergence means the template has turn-dependent state
```

A divergence here is a design hint, not a bug: it says the template cannot be used for streaming / piecewise packing without care.

## 3. Validation steps

A complete audit report should show:

1. **Tokenizer fingerprint.** Class name, vocab size, BOS/EOS/PAD/UNK ids, padding side, `is_fast`.
2. **Special token table.** All added tokens with id, content, `special`, `single_word`.
3. **Chat template rendering.** The full Jinja string and a rendered example for `[system, user, assistant, user]`, with `add_generation_prompt=True` and `=False`.
4. **Label mask proof.** `(token, label)` table for one example, with assistant-only tokens carrying loss.
5. **Round-trip check.** Pass/fail on at least five strings including emoji, accented chars, code, mixed whitespace, and the empty string.
6. **Boundary checks.** Leading-space behavior and whitespace-sensitivity note.
7. **PAD/EOS policy.** Explicit statement of which id pads, which id ends a turn, which id ends the document, and the per-turn stop criterion for generation.
8. **(If training) Decoder round-trip.** Encode → decode → re-encode for a chat prompt and show the two id sequences are equal.

Report format can be free-form markdown; the grader is checking that every item appears.

## 4. Rubric

10 points total.

| Dimension | Points | Fail (0) | Partial | Full |
|---|---|---|---|---|
| Tokenizer inventory present | 1 | — | partial fields | all fields from §3.1 |
| Special tokens inventoried | 1 | — | missing EOT/EOS distinction | full table + EOT/EOS distinction called out |
| Chat template dumped & rendered | 2 | not shown | shown but `add_generation_prompt` untested | both variants, trailing-newline observed |
| Label mask derived from template | 3 | regex-on-string approach | structural but masks wrong tokens | assistant-tokens-only; EOT-on-assistant policy stated; `-100` used |
| Encoding-boundary checks | 1 | none | one | round-trip + leading-space + at least one Unicode edge |
| PAD/EOS policy statement | 1 | missing or wrong | partial | PAD set, EOT = per-turn stop, EOS = doc stop, all stated |
| Validation evidence | 1 | hand-wavy | one check shown | at least three §3 checks shown as code output |

Pass threshold: 7/10, with "label mask derived from template" ≥ 2.

## 5. Common mistakes

- **Regex-masking the prompt instead of using the template structure.** A template change (extra space, new header format) silently breaks regex masks; the `return_assistant_tokens_mask` path tracks the template [HF-CT].
- **Using `eos_token_id` as the per-turn stop for Llama 3.** The per-turn stop is `<|eot_id|>`. `<|end_of_text|>` is the document stop. These are different ids.
- **Leaving `pad_token_id=None` and letting the collator reuse EOS.** Produces plausible-looking loss curves while scrambling the stop signal.
- **Right-padding at decode time.** The last real token is now in different columns for short rows; generation starts from the pad column.
- **Resizing the embedding to add a new role token when a `<|reserved_special_token_N|>` exists.** Avoid untrained embedding rows unless there is no other choice.
- **Tokenizing the full conversation without the template.** Hand-concatenating role markers diverges from the base model's convention on whitespace, newlines, or token boundaries.
- **Mixing Llama 2 and Llama 3 templates.** They are completely different (`[INST] ... [/INST]` vs header-id framing). A model trained with the wrong family's template will not stop and will not role-switch.
- **Treating `tokenizer.model_max_length` as the model's real max context.** It is the tokenizer's declared limit; always cross-check `model.config.max_position_embeddings`.

## 6. References

- [HF-Tok] HuggingFace Transformers, Tokenizer documentation. <https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer>
- [HF-CT] HuggingFace Transformers, Chat templating guide. <https://huggingface.co/docs/transformers/main/en/chat_templating>
- [HF-TOK-LIB] HuggingFace Tokenizers (Rust-backed fast tokenizers). <https://huggingface.co/docs/tokenizers>
- [L3] Meta, "The Llama 3 Herd of Models," 2024. <https://arxiv.org/abs/2407.21783>
- [BPE] Sennrich, Haddow, Birch, "Neural Machine Translation of Rare Words with Subword Units" (BPE), 2016. <https://arxiv.org/abs/1508.07909>
- [SP] Kudo & Richardson, "SentencePiece: A simple and language independent subword tokenizer," 2018. <https://arxiv.org/abs/1808.06226>
- Jinja2 template language. <https://jinja.palletsprojects.com/>
