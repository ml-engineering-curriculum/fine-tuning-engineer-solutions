# mod-103 · exercise-02 — Masking and Packing Correctness — Solution

## 1. Solution overview

Packing and masking break the same invariant from two directions. Packing lets independent conversations share one `input_ids` sequence to raise GPU utilization; masking decides which of those positions contribute to the loss. When either is wrong, the loss looks healthy and the model learns the wrong thing — attention across document boundaries, loss on pad tokens, loss on user turns. This is the single most expensive category of SFT bug because nothing crashes.

The exercise codifies four tests the learner must pass on a packed, loss-masked batch:

- **Loss is assistant-only, terminator-included.** Every `label != -100` position corresponds to assistant content or its turn terminator.
- **Attention never crosses a document boundary.** Position *t* in sequence *j* never attends to position *t'* that belongs to a different packed document.
- **Position IDs restart at every boundary.** Document *k*'s first token has `position_ids == 0`, not the running index in the packed row.
- **Pads never enter the loss.** Padding tokens anywhere in the batch have `label == -100`.

Modern stacks (TRL ≥ 0.12 + `transformers` ≥ 4.44 + FlashAttention-2 ≥ 2.5 with `flash_attn_varlen_func`, or SDPA + `attention_mask` as a document-id tensor) give you all four if configured correctly [TRL] [HFTPL] [FA2]. The exercise's point is to prove it, not take it on faith.

The worked example uses `meta-llama/Llama-3.2-1B` and five hand-picked multi-turn conversations from `HuggingFaceH4/no_robots` — few enough to read every position by hand, varied enough to catch edge cases (short/long, 2-turn/5-turn, assistant-ends-with-code-fence).

## 2. Worked implementation

### 2.1 Build the batch three ways

The comparison is the pedagogy. The learner constructs the same five conversations under three configurations:

| Config | Packing | Loss mask | Expected outcome |
|---|---|---|---|
| A | off | assistant-only | Correct baseline. One example per row, long tail of pads, loss honest. |
| B | on, no doc mask | assistant-only | Fast per step but wrong — attention leaks across boundaries; model learns phantom cross-doc dependencies. |
| C | on, with doc mask | assistant-only | Correct and fast. Boundary enforced by `attention_mask` / `position_ids` / FA-varlen. |

Only Config C goes to the trainer. A and B exist to make the test suite fire.

### 2.2 `pack.py` — reference packer with document ids

```python
# pack.py
from __future__ import annotations
import torch
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100

def render_one(tokenizer: PreTrainedTokenizerBase, messages: list[dict]) -> tuple[list[int], list[int], list[int]]:
    """Return (input_ids, labels, assistant_mask) for a single conversation using the
    chat template. `assistant_mask` is the tokenizer's own 0/1 mask — kept alongside
    labels so invariants can cross-check membership against an independent source.
    """
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_assistant_tokens_mask=True,
    )
    ids, amask = enc["input_ids"], enc["assistant_masks"]
    labels = [tid if m == 1 else IGNORE_INDEX for tid, m in zip(ids, amask)]
    return ids, labels, list(amask)

def pack_examples(tokenizer, conversations, max_seq_length: int):
    """Greedy first-fit decreasing packer. Returns a dict suitable for an FA-varlen
    attention path OR an SDPA 4D-mask attention path.

    Output keys:
      input_ids       : (B, S) int64
      labels          : (B, S) int64
      attention_mask  : (B, S) int64  (1=real, 0=pad)
      position_ids    : (B, S) int64, restarts at each doc boundary
      cu_seqlens      : (B, ...)-shaped python list of 1D int32 tensors; lengths of
                        each packed doc per row, for flash_attn_varlen_func.
    """
    rendered = [render_one(tokenizer, c) for c in conversations]
    # Drop oversize; keep a counter the caller can log.
    rendered = [(ids, lab, am) for ids, lab, am in rendered if len(ids) <= max_seq_length]
    # FFD: longest first.
    rendered.sort(key=lambda r: -len(r[0]))

    rows = []           # list[list[(ids, labels, assistant_mask)]]
    for ids, lab, amask in rendered:
        placed = False
        for row in rows:
            if sum(len(x[0]) for x in row) + len(ids) <= max_seq_length:
                row.append((ids, lab, amask)); placed = True; break
        if not placed:
            rows.append([(ids, lab, amask)])

    pad_id   = tokenizer.pad_token_id
    input_ids_batch, labels_batch, am_batch, pid_batch, asst_batch, cus = [], [], [], [], [], []
    for row in rows:
        ids = sum((x[0] for x in row), [])
        lab = sum((x[1] for x in row), [])
        asst = sum((x[2] for x in row), [])
        # position_ids restart at every doc:
        pids = sum((list(range(len(x[0]))) for x in row), [])
        # per-doc lengths for FA-varlen:
        lens = [len(x[0]) for x in row]

        pad_n = max_seq_length - len(ids)
        if pad_n > 0:
            ids  = ids  + [pad_id] * pad_n
            lab  = lab  + [IGNORE_INDEX] * pad_n
            asst = asst + [0] * pad_n
            pids = pids + [0] * pad_n          # value doesn't matter; mask will zero it
            # NOTE: the pad run is NOT a new document. It never contributes to loss
            # (labels are -100) and in the FA-varlen path we do not include it in
            # cu_seqlens; the forward skips the pad region entirely.
        am = [1] * (max_seq_length - pad_n) + [0] * pad_n

        input_ids_batch.append(ids)
        labels_batch.append(lab)
        am_batch.append(am)
        pid_batch.append(pids)
        asst_batch.append(asst)
        # cu_seqlens: cumulative prefix with a leading 0, int32, FA-2 expects it.
        cu = [0]
        for L in lens: cu.append(cu[-1] + L)
        cus.append(torch.tensor(cu, dtype=torch.int32))

    return {
        "input_ids":      torch.tensor(input_ids_batch, dtype=torch.long),
        "labels":         torch.tensor(labels_batch,    dtype=torch.long),
        "attention_mask": torch.tensor(am_batch,        dtype=torch.long),
        "position_ids":   torch.tensor(pid_batch,       dtype=torch.long),
        "assistant_mask": torch.tensor(asst_batch,      dtype=torch.long),
        "cu_seqlens":     cus,
    }
```

Notes:

- The packer never splits a conversation across two rows. Splitting is legal in continued pretraining but wrong for SFT: a half-conversation starting mid-assistant turn teaches the model to answer without a user turn.
- `position_ids` are the signal RoPE actually sees. Restart at every document boundary or the model learns positional artifacts tying distant documents together.
- `attention_mask` here is the *padding* mask. Document-boundary attention is enforced by one of two routes: (a) the FA-2 `flash_attn_varlen_func` path, which uses `cu_seqlens` directly and never attends across them; or (b) an SDPA path with a 4-D `(B, 1, S, S)` additive mask that puts `-inf` on all off-document entries. Both are correct; FA-2 is dramatically faster and uses no `S²` memory.

### 2.3 Trainer wiring (SDPA fallback)

If FlashAttention-2 is not available, build the per-row document mask once and pass it through a custom collator:

```python
def build_doc_attn_mask(cu_seqlens_1d: torch.Tensor, S: int) -> torch.Tensor:
    """Returns a (1, S, S) boolean matrix, True where attention is ALLOWED.
    Within each doc, lower-triangular; across docs, False."""
    mask = torch.zeros(S, S, dtype=torch.bool)
    for i in range(len(cu_seqlens_1d) - 1):
        a, b = int(cu_seqlens_1d[i]), int(cu_seqlens_1d[i+1])
        tri = torch.tril(torch.ones(b - a, b - a, dtype=torch.bool))
        mask[a:b, a:b] = tri
    return mask.unsqueeze(0)
```

Pass as `attention_mask` to `model(**batch)` when using `attn_implementation="sdpa"`; `transformers` accepts a 4-D mask and treats `False` as `-inf` [HFTPL].

### 2.4 `invariants.py` — the four checks

```python
# invariants.py
import torch
IGNORE_INDEX = -100

def _assistant_spans_from_markers(tokenizer, ids_row, attn_row):
    """Independent ground truth for assistant membership: scan the token stream
    for role-header patterns and return (start, end) half-open spans covering
    each assistant turn's content plus its terminator. Does NOT consult
    `return_assistant_tokens_mask`, so a tokenizer that emits an all-zero,
    all-one, or shifted mask still yields correct spans here. Supports Llama-3
    (`<|start_header_id|>assistant<|end_header_id|> ... <|eot_id|>`) and ChatML
    (`<|im_start|>assistant ... <|im_end|>`). Returns None when neither family's
    special tokens are in the tokenizer vocabulary."""
    def _tid(tok):
        i = tokenizer.convert_tokens_to_ids(tok)
        return None if i is None or i == tokenizer.unk_token_id else i

    ids, attn = ids_row.tolist(), attn_row.tolist()
    role = tokenizer.encode("assistant", add_special_tokens=False)
    N, R = len(ids), len(role)
    spans, i = [], 0

    s_hdr, e_hdr, eot = _tid("<|start_header_id|>"), _tid("<|end_header_id|>"), _tid("<|eot_id|>")
    if None not in (s_hdr, e_hdr, eot):
        while i + R + 2 < N:
            if (attn[i] and ids[i] == s_hdr
                and ids[i+1:i+1+R] == role and ids[i+1+R] == e_hdr):
                start = i + 2 + R
                j = start
                while j < N and attn[j] and ids[j] != eot:
                    j += 1
                if j < N and ids[j] == eot:
                    spans.append((start, j + 1)); i = j + 1; continue
            i += 1
        return spans

    im_start, im_end = _tid("<|im_start|>"), _tid("<|im_end|>")
    if None not in (im_start, im_end):
        while i + R + 1 < N:
            if attn[i] and ids[i] == im_start and ids[i+1:i+1+R] == role:
                start = i + 1 + R
                j = start
                while j < N and attn[j] and ids[j] != im_end:
                    j += 1
                if j < N and ids[j] == im_end:
                    spans.append((start, j + 1)); i = j + 1; continue
            i += 1
        return spans

    return None

def assert_label_mask_assistant_only(tokenizer, batch):
    """Label mask must agree with TWO independent sources of truth:

    1. The tokenizer's own assistant-token mask (`batch['assistant_mask']`,
       produced by `apply_chat_template(..., return_assistant_tokens_mask=True)`).
       Catches intra-row label corruption.
    2. Assistant spans re-derived from the role-marker tokens in `input_ids`
       itself via `_assistant_spans_from_markers`. Catches an all-zero, all-one,
       or shifted `assistant_mask` — which (1) alone would accept tautologically
       because labels and assistant_mask share one source.

    See pack.py §2.2 for where `assistant_mask` originates."""
    ids, lab = batch["input_ids"], batch["labels"]
    assert "assistant_mask" in batch, (
        "batch is missing 'assistant_mask'; packer must emit the tokenizer's own "
        "assistant-token mask alongside labels so this invariant has a second source")
    asst = batch["assistant_mask"]
    pad_tensor = batch["attention_mask"] == 0  # pad positions

    total_trainable = 0
    for row_i in range(ids.shape[0]):
        for t in range(ids.shape[1]):
            is_pad = bool(pad_tensor[row_i, t].item())
            want_trainable = (asst[row_i, t].item() == 1) and not is_pad
            is_trainable = lab[row_i, t].item() != IGNORE_INDEX
            assert want_trainable == is_trainable, (
                f"row {row_i} pos {t}: assistant_mask={int(asst[row_i, t])} "
                f"pad={is_pad} but labels={'trainable' if is_trainable else 'masked'}")
            if is_trainable:
                assert lab[row_i, t].item() == ids[row_i, t].item(), \
                    f"row {row_i} pos {t}: label != input"
                total_trainable += 1
    assert total_trainable > 0, (
        "no trainable positions in this batch — the loss would be NaN and the "
        "invariant would still trivially pass without this guard")

    # Independent cross-check against role markers embedded in input_ids. This
    # uses a DIFFERENT code path than return_assistant_tokens_mask, so it still
    # fires when the tokenizer's mask is all-zero / all-one / off-by-one.
    cross_checked = False
    for row_i in range(ids.shape[0]):
        spans = _assistant_spans_from_markers(
            tokenizer, ids[row_i], batch["attention_mask"][row_i])
        if spans is None:
            continue
        cross_checked = True
        want = {t for s, e in spans for t in range(s, e)}
        got = {t for t in range(ids.shape[1])
               if asst[row_i, t].item() == 1 and not bool(pad_tensor[row_i, t].item())}
        assert want == got, (
            f"row {row_i}: role-marker-derived assistant positions disagree with "
            f"assistant_mask ({len(want ^ got)} positions differ) — label mask cannot be trusted")
        for s, e in spans:
            # The terminator sits at e-1 by construction; it must be trainable
            # so the model learns to stop even when a next packed doc follows.
            assert lab[row_i, e - 1].item() != IGNORE_INDEX, (
                f"row {row_i} pos {e-1}: terminator of assistant span [{s},{e}) is masked")
    assert cross_checked, (
        "no row admitted a role-marker cross-check — this invariant would degenerate "
        "to the tokenizer-vs-itself tautology. Use a Llama-3 or ChatML template, or "
        "extend _assistant_spans_from_markers for the template in use")

def assert_pads_masked(tokenizer, batch):
    ids, lab = batch["input_ids"], batch["labels"]
    pad = tokenizer.pad_token_id
    assert pad is not None, "tokenizer has no pad token"
    for row_i in range(ids.shape[0]):
        for t in range(ids.shape[1]):
            if ids[row_i, t].item() == pad:
                assert lab[row_i, t].item() == IGNORE_INDEX, \
                    f"row {row_i} pos {t}: pad has active label"

def assert_position_ids_restart(batch):
    pids = batch["position_ids"]
    for row_i in range(pids.shape[0]):
        cu = batch["cu_seqlens"][row_i].tolist()
        for a, b in zip(cu, cu[1:]):
            # positions [a, b) must be 0..b-a-1
            want = torch.arange(b - a, dtype=pids.dtype)
            got = pids[row_i, a:b]
            assert torch.equal(got, want), (
                f"row {row_i} doc[{a}:{b}] position_ids={got.tolist()} != {want.tolist()}")

def assert_doc_attention_no_cross_leak(model, batch, device):
    """Gradient-based leakage test: perturb token t of doc A; check that logits at
    positions in doc B are unchanged. Unchanged = correct isolation."""
    model.eval().to(device)
    b = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        base = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                     position_ids=b["position_ids"]).logits
    row_i = 0
    cu = batch["cu_seqlens"][row_i].tolist()
    assert len(cu) >= 3, "need at least two packed docs in row 0 for this test"
    doc_a = (cu[0], cu[1]); doc_b = (cu[1], cu[2])
    # flip one token near the end of doc A to something benign (e.g., 0):
    poked = b["input_ids"].clone()
    poked[row_i, doc_a[1] - 1] = 0
    with torch.no_grad():
        alt = model(input_ids=poked, attention_mask=b["attention_mask"],
                    position_ids=b["position_ids"]).logits
    diff_in_b = (alt[row_i, doc_b[0]:doc_b[1]] - base[row_i, doc_b[0]:doc_b[1]]).abs().max()
    assert diff_in_b.item() < 1e-4, (
        f"cross-doc leak: perturbing doc-A changed doc-B logits by {diff_in_b:.3e}")
```

Invariant 4 is the sharp one. If attention leaks across documents, perturbing a token in doc A changes the logits in doc B. If isolation holds, those logits are bit-identical (up to numerical noise).

### 2.5 Expected outputs — the three configs, side by side

Run each invariant against Configs A, B, C and tabulate:

| Invariant | Config A (no packing) | Config B (packed, no doc mask) | Config C (packed, doc mask) |
|---|---|---|---|
| label ↔ assistant | PASS | PASS (same labels) | PASS |
| pads masked | PASS | PASS | PASS |
| position_ids restart | n/a (single doc per row) | FAIL — pids run 0..S-1 | PASS |
| no cross-doc leak | PASS (no cross docs to leak) | FAIL — doc A perturbation shifts doc B logits by ~1e-1 | PASS |

Config B is the realistic bug: it passes label and pad checks and prints a loss curve that looks fine. The trainer sees a lower loss per wall-clock-second because each row is dense with assistant tokens; the model sees a corrupt causal graph.

### 2.6 Tiny visualization: boundary grid

```python
def print_doc_boundary_grid(cu_seqlens_1d, labels_row, tok, input_ids_row):
    boundaries = cu_seqlens_1d.tolist()
    docs = [(a, b) for a, b in zip(boundaries, boundaries[1:])]
    print(" idx  doc  tok                        label")
    for t, (tid, lab) in enumerate(zip(input_ids_row.tolist(), labels_row.tolist())):
        d = next((i for i, (a, b) in enumerate(docs) if a <= t < b), -1)
        marker = "train" if lab != -100 else "mask"
        print(f"{t:>4}  {d:>3}  {tok.decode([tid])!r:<28}  {marker}")
```

A reviewer should be able to scroll this table and name, for every position, which document it belongs to and whether the loss sees it. If they cannot, the pipeline is not auditable.

### 2.7 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| FA-2 varlen path | `cu_seqlens` isolates docs with no S² mask memory; fastest correct path | SDPA + 4-D mask | SDPA 4-D mask works everywhere, costs S² memory; pick it when FA-2 unavailable |
| Pack but never split a conversation | Splitting teaches the model to start a response without a prompt | Split with `wrap=True` | Split is fine for continued pretraining, not SFT |
| Restart `position_ids` per doc | RoPE phase otherwise ties unrelated documents at the same relative distance | Monotone pids 0..S-1 | Monotone looks simpler and is incorrect |
| Keep terminator in loss, inside a packed row | Model must learn to stop even when a next document follows | Mask terminator | Mask-terminator models over-generate at inference |
| `IGNORE_INDEX = -100` | PyTorch `CrossEntropyLoss(ignore_index=-100)` default | Any negative int | Deviating from the default breaks unmodified `Trainer` losses |
| Use `return_assistant_tokens_mask=True` | Tokenizer-maintained ground truth | Hand-rolled mask | Hand-rolled drifts when templates update (same hazard as mod-102 ex-03) |

## 3. Validation steps

1. **Run all four invariants on Configs A, B, C.** Record which pass/fail. Expected: B fails `position_ids_restart` and `no_cross_doc_leak`. If B passes everything, the test is not fire-sensitive — likely the "perturbation" token lives past the model's effective receptive field.
2. **Loss floor sanity.** Compute the loss of each config on a batch where every assistant token is perfectly predicted; it must equal zero (within fp32 rounding). If not, labels include non-assistant tokens that the model cannot predict, inflating the floor.
3. **Decode a packed row end-to-end.** `tokenizer.decode(input_ids[0][attention_mask[0].bool()])` must concatenate exactly the rendered strings of the packed conversations. No stray BOS between docs, no missing terminator.
4. **Doc counts tie out.** `sum(cu_seqlens[-1] for cu_seqlens in batch["cu_seqlens"]) == (batch["attention_mask"] == 1).sum()`. The packed-and-counted tokens equal the non-pad tokens.
5. **Causality within doc.** For a doc of length *L* inside a row, the token at position *k* (global index *a+k*) attends only to positions in `[a, a+k]`. Verify by perturbing position *a+k+1* and asserting logits at *a+k* unchanged.
6. **Loss parity.** Config A and Config C on the same five conversations must produce the same per-conversation token-level loss, up to numerical precision. If Config C's loss is systematically lower, it is leaking attention from future packed docs.
7. **Regression test on a template update.** Pin the exact `transformers` and `trl` versions. Re-render the same five conversations after an upgrade; assert byte-identical `input_ids` and `labels`. Chat templates drift between library versions and this is how silent label bugs ride in.

## 4. Rubric for grading the learner's packing audit

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Packer correctness** | Splits conversations mid-turn, or does not restart position_ids | Packs without splitting but position_ids are monotone | Packs without splitting, position_ids restart, `cu_seqlens` emitted |
| **Loss masking** | Trains on user/system, or terminator is masked | Assistant-only but terminator excluded, or pads included | Assistant-only, terminator included, pads masked to `-100` |
| **Cross-doc isolation** | No test; "FlashAttention handles it" is the whole answer | Test exists but only on one sequence length / one layout | Perturbation test runs on at least 2 configs; Config B is demonstrated to fail |
| **Config comparison** | Only runs Config C | A and C compared; B omitted | All three run; results tabled; B's symptoms described |
| **Report + reproduce** | Ad-hoc output, no command to rerun | Report exists but no invariant list / no diff of configs | Report lists all four invariants, pass/fail matrix, and a `python -m invariants` command |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Trusting that `packing=True` on a stock trainer implies boundary attention. Several stacks pack but attend globally — read the attention path, not the config flag. TRL's `SFTTrainer` with `packing=True` routes through the dataset collator; verify it emits `position_ids` restart and uses FA-varlen or a 4-D mask [TRL].
- Masking the terminator by taking "assistant content" to mean "text the assistant said" rather than "assistant tokens including stop." Models trained this way never stop at inference.
- Using `IGNORE_INDEX = 0`. 0 is `<unk>` or a real token in most tokenizers — the loss silently trains on pad.
- Testing cross-doc leakage with a change at the end of doc B instead of a change inside doc A. The attention path is causal; changes to the *right* of a position never affect it. The flip must be in an earlier doc.
- Packing eval. Eval loss is a per-example average; packing turns it into a per-row average weighted by packed length. The number stops being comparable to unpacked baselines. Keep eval unpacked.
- Reporting `train_loss` only. Packing makes `train_loss` go down because the average absorbs more assistant tokens per step; it does not mean the model got better.
- Padding to the right and then rolling the pads into position_ids. Pads must sit *outside* the `cu_seqlens` structure; otherwise the model trains with a "phantom doc" at the tail.
- Not pinning `transformers` / `trl` / `flash-attn` versions. Chat templates and packing semantics change between versions; audits become non-reproducible overnight.

## 6. References

- [TRL] HuggingFace TRL — `SFTTrainer` + packing semantics. <https://huggingface.co/docs/trl/main/en/sft_trainer>
- [HFTPL] HuggingFace chat templating (`return_assistant_tokens_mask`) and `transformers` 4-D attention mask docs. <https://huggingface.co/docs/transformers/main/en/chat_templating>
- [FA2] Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023 — `flash_attn_varlen_func` interface. <https://arxiv.org/abs/2307.08691> / <https://github.com/Dao-AILab/flash-attention>
- [L3] Meta, "The Llama 3 Herd of Models" (special tokens incl. `<|eot_id|>`), 2024. <https://arxiv.org/abs/2407.21783>
- [AX] Axolotl `sample_packing` + `pad_to_sequence_len` docs. <https://github.com/OpenAccess-AI-Collective/axolotl>
- [GPT] OpenAI "GPT-NeoX / Megatron" packing discussion — document separators in packed sequences. <https://github.com/EleutherAI/gpt-neox>
- OWASP Machine Learning Security Top 10 — ML-02 Data Poisoning context (silent label flips are a within-pipeline poisoning vector). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — "Poison Training Data" technique and ML supply-chain pattern (library version drift). <https://atlas.mitre.org/>
- NIST AI Risk Management Framework — Measure 2.3 (test the system as deployed, not as idealized). <https://www.nist.gov/itl/ai-risk-management-framework>
