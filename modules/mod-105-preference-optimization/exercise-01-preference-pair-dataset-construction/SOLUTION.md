# mod-105 · exercise-01 — Preference Pair Dataset Construction — Solution

## 1. Solution overview

The deliverable is not just a `.jsonl` of `(prompt, chosen, rejected)` rows. It is a reproducible pipeline that an auditor can run on raw sources and get the same preference pairs back, with a per-row provenance trail explaining why `chosen` is preferred to `rejected`. A reviewer should be able to read `collection_plan.md`, `pairs_pipeline.py`, `pairs.jsonl`, `drops/`, `reliability_report.md`, and `data_card.md` and reconstruct every pair's path from raw to shipped.

Three ideas carry the exercise:

- **Preference data is labels, not just text.** The pair's value lives in the comparison signal — who preferred `chosen` over `rejected`, on what axis, and with how much agreement. A pair with no meta is a pair you can't debug when the reward model learns the wrong thing.
- **"Preference" is not one thing.** Helpfulness, harmlessness, honesty, and style are different signals that train different reward models. Mixing them silently is the single most common mistake downstream; a `preference_dimension` label per row is non-negotiable.
- **Inter-annotator agreement is the data card's headline number.** If annotators (human or model-as-judge) disagree more than 30 % on a slice, the reward model will fit the noise, and PPO (exercise-03) or DPO (exercise-04) will optimize an arbitrary signal. Measuring agreement is the audit, not an afterthought.

This walkthrough assembles ~5 000 pairs from three public sources — `Anthropic/hh-rlhf` [HH22], `argilla/ultrafeedback-binarized-preferences` [UFB23], and `OpenAssistant/oasst2` rankings [OASST] — plus a small model-generated slice to show the end-to-end flow. The structure generalizes to any domain where you can elicit comparisons.

## 2. Worked implementation

### 2.1 `collection_plan.md` (write this first)

Fill concrete thresholds; every filter and every annotation decision later should trace back here.

```markdown
# Preference Pair Collection Plan — v1

## Scope
- Target: ~5,000 pairs, English, single-turn + short multi-turn (≤ 4 turns).
- Dimensions trained in this pass: helpfulness, harmlessness.
- Out of scope for v1: style-only pairs, non-EN, tool-use traces, >4096-token prompts.

## Sources (and their preference semantics)

| Source | Elicitation | Native dimension | License | Our use |
|---|---|---|---|---|
| Anthropic/hh-rlhf (`helpful-base`) | 2-way human comparison, open-ended | helpfulness | MIT | helpfulness pool |
| Anthropic/hh-rlhf (`harmless-base`) | 2-way human comparison, red-team prompts | harmlessness | MIT | harmlessness pool |
| argilla/ultrafeedback-binarized | GPT-4 pointwise scores → binarized | helpfulness + honesty | MIT | helpfulness pool |
| OpenAssistant/oasst2 (rankings) | Human rank over N replies in a thread | mixed (helpful + style) | Apache-2.0 | helpfulness pool |
| Synthetic (ours) | Two completions from a single SFT base, judged by a stronger model | helpfulness | derivative | audit sample only |

## Preference dimensions
Every pair carries exactly one `preference_dimension` in {`helpful`, `harmless`}.
Rows that would qualify on more than one dimension are *duplicated* (one copy
per dimension) only if the judgment for each dimension is independently
recorded. Otherwise, assign the dimension that drove the elicitation.

## Agreement protocol
- Human-sourced: use the source's native agreement when available (HH ships a
  single preference; UFB ships 4 pointwise scores we binarize; OASST ships
  explicit ranks). Trust-weight each source (see `reliability_report.md`).
- Model-as-judge sample: for the synthetic slice, judge each pair with two
  independent prompt templates on a strong model; keep only pairs where both
  judgments agree on the same winner. Discard ties.

## Quality filters (applied before pair construction)
1. Structural: well-formed turns, non-empty assistant content, roles alternate.
2. Language: English on both prompt and both responses (fastText `lid.176`, p ≥ 0.80).
3. Length: prompt ∈ [16, 2048] tokens, each response ∈ [8, 2048] tokens, pair total ≤ 4096 tokens under the target tokenizer (`meta-llama/Meta-Llama-3.1-8B-Instruct`).
4. PII: drop rows with Presidio hits on {EMAIL, PHONE, CREDIT_CARD, SSN, IBAN, IP}.
5. Near-duplicate: drop pairs whose `(prompt, chosen)` or `(prompt, rejected)` MinHash-Jaccard ≥ 0.9 against anything already in the pool or against our SFT training set (links to `mod-102 exercise-02 decontam`).
6. Trivial pairs: drop if `chosen` and `rejected` are identical, or if Jaccard ≥ 0.95 on the response tokens; the reward model learns nothing from these.
7. Response-quality floor: reject pairs where BOTH responses fail basic SFT quality rules from `mod-102` (repetition, teacher artifacts, mid-sentence truncation).

## Pair construction
- Raw labeled pair from source → single `pairs.jsonl` row (see schema below).
- Ranked list of N responses (OASST) → generate `C(N, 2)` pairs with
  `chosen = higher-ranked`, `rejected = lower-ranked`, and tag `tie_rank_delta`
  = the rank-position difference. We drop pairs with `tie_rank_delta < 2` to
  avoid near-tie noise.
- Pointwise scores (UFB) → binarize by picking the highest-scored response as
  `chosen` and a uniformly-sampled strictly-lower-scored response as
  `rejected`. Require score margin ≥ 1.0 on UFB's 1–10 scale.

## Row schema (for pairs.jsonl)

```json
{
  "id": "sha1(prompt+chosen+rejected+source)",
  "source": "hh-rlhf:helpful-base",
  "source_row_id": "orig-id-or-hash",
  "license": "MIT",
  "preference_dimension": "helpful",
  "prompt": "...",                          // rendered via tokenizer.apply_chat_template, decoded back
  "chosen": "...",
  "rejected": "...",
  "chosen_rank_meta": {"rank": 1},          // or {"score": 8.5}
  "rejected_rank_meta": {"rank": 3},        // or {"score": 4.0}
  "agreement": {
    "kind": "single_rater" | "ranking" | "pointwise_binarized" | "two_judge_agreement",
    "value": 1.0,                            // {0.0, 0.5, 1.0} for binary; score margin for pointwise
    "n_raters": 1
  },
  "length_tokens": {"prompt": 184, "chosen": 312, "rejected": 298},
  "safety_flags": [],
  "decontam": {"mod_102_ex_02": "pass"},
  "pipeline_version": "v1"
}
```

## Out-of-pool but kept
- `safety_refusals_rejected.jsonl` — pairs where the *rejected* response is an
  appropriate safety refusal and the *chosen* is an unsafe answer. These are
  **re-polarized** (swap chosen/rejected) and routed to the harmlessness pool.
  Never discard them; they are the most valuable harmlessness signal.
- `red_team_only.jsonl` — red-team prompts without a clean safe pair; kept
  for exercise-02 (reward model) negative sampling if needed.

## Splits
- 85 / 10 / 5 train / val / test.
- Stratify by `(source, preference_dimension)`.
- Keep test set private to this exercise until the module ends; never show it
  to the reward model or the policy in exercise-02/03/04.
```

### 2.2 `pairs_pipeline.py` — reference skeleton

```python
# pairs_pipeline.py — illustrative skeleton; trimmed for brevity, not correctness.
from __future__ import annotations
import hashlib, itertools, json, pathlib, random, re
from dataclasses import dataclass, asdict
from typing import Iterator

from datasets import load_dataset
from transformers import AutoTokenizer
from datasketch import MinHash, MinHashLSH
import fasttext
from presidio_analyzer import AnalyzerEngine

OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
DROPS = OUT / "drops"; DROPS.mkdir(exist_ok=True)

TOK = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
LID = fasttext.load_model("lid.176.bin")
PII = AnalyzerEngine()

PII_TYPES = {"EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
             "US_SSN", "IBAN_CODE", "IP_ADDRESS"}

RNG = random.Random(0)                  # deterministic sampling anywhere we sample


def sha1(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8")); h.update(b"\x1f")
    return h.hexdigest()


def tok_len(text: str) -> int:
    return len(TOK.encode(text, add_special_tokens=False))


def is_english(text: str, p_min: float = 0.80) -> bool:
    labels, probs = LID.predict(text.replace("\n", " ")[:4000], k=1)
    return labels[0] == "__label__en" and probs[0] >= p_min


def pii_hit(text: str) -> list[str]:
    return [r.entity_type for r in PII.analyze(text=text, language="en")
            if r.entity_type in PII_TYPES]


def minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for tok in re.findall(r"\w+", text.lower()):
        m.update(tok.encode("utf-8"))
    return m


@dataclass
class Pair:
    id: str
    source: str
    source_row_id: str
    license: str
    preference_dimension: str          # "helpful" | "harmless"
    prompt: str
    chosen: str
    rejected: str
    chosen_rank_meta: dict
    rejected_rank_meta: dict
    agreement: dict
    length_tokens: dict
    safety_flags: list
    decontam: dict
    pipeline_version: str = "v1"


def filter_and_yield(pair: Pair, lsh: MinHashLSH,
                     drop_log: dict[str, list[str]]) -> Pair | None:
    """Apply plan's filters in cheap-to-expensive order."""
    reason: list[str] = []

    # Structural
    if not pair.prompt.strip() or not pair.chosen.strip() or not pair.rejected.strip():
        reason.append("structural:empty")
    if pair.chosen.strip() == pair.rejected.strip():
        reason.append("trivial:identical")

    # Length
    L = {
        "prompt":   tok_len(pair.prompt),
        "chosen":   tok_len(pair.chosen),
        "rejected": tok_len(pair.rejected),
    }
    if L["prompt"] < 16 or L["prompt"] > 2048: reason.append("length:prompt")
    if L["chosen"] < 8 or L["chosen"] > 2048:  reason.append("length:chosen")
    if L["rejected"] < 8 or L["rejected"] > 2048: reason.append("length:rejected")
    if L["prompt"] + max(L["chosen"], L["rejected"]) > 4096: reason.append("length:pair_total")

    # Language (cheap enough to run here)
    if not is_english(pair.prompt):   reason.append("lang:prompt")
    if not is_english(pair.chosen):   reason.append("lang:chosen")
    if not is_english(pair.rejected): reason.append("lang:rejected")

    # PII
    for field in ("prompt", "chosen", "rejected"):
        hits = pii_hit(getattr(pair, field))
        if hits:
            reason.append(f"pii:{field}:{','.join(hits)}")

    if reason:
        drop_log.setdefault("|".join(reason), []).append(pair.id)
        return None

    # Near-duplicate (vs. things we've already admitted)
    m_c = minhash(pair.prompt + "\n" + pair.chosen)
    m_r = minhash(pair.prompt + "\n" + pair.rejected)
    if lsh.query(m_c) or lsh.query(m_r):
        drop_log.setdefault("dedup", []).append(pair.id)
        return None
    lsh.insert(pair.id + ":c", m_c)
    lsh.insert(pair.id + ":r", m_r)

    pair.length_tokens = L
    return pair


# ---- source adapters --------------------------------------------------------

def from_hh(split: str, subset: str, dim: str) -> Iterator[Pair]:
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
    for i, row in enumerate(ds):
        # hh-rlhf ships `chosen` and `rejected` as full dialogs. Split off
        # the final assistant turn as the response, prior turns as prompt.
        prompt_c, resp_c = _hh_split(row["chosen"])
        prompt_r, resp_r = _hh_split(row["rejected"])
        if prompt_c != prompt_r:
            # Shouldn't happen in hh-rlhf; skip if the dataset shifts.
            continue
        yield Pair(
            id=sha1(prompt_c, resp_c, resp_r, "hh-rlhf"),
            source=f"hh-rlhf:{subset}",
            source_row_id=f"row:{i}",
            license="MIT",
            preference_dimension=dim,
            prompt=prompt_c,
            chosen=resp_c,
            rejected=resp_r,
            chosen_rank_meta={"rank": 1},
            rejected_rank_meta={"rank": 2},
            agreement={"kind": "single_rater", "value": 1.0, "n_raters": 1},
            length_tokens={},                  # filled later
            safety_flags=[],
            decontam={"mod_102_ex_02": "pending"},
        )


def _hh_split(dialog: str) -> tuple[str, str]:
    # hh-rlhf dialog format: "\n\nHuman: ...\n\nAssistant: ..." repeated.
    parts = re.split(r"\n\n(?=(?:Human|Assistant):)", dialog.strip())
    if parts and parts[0] == "":
        parts = parts[1:]
    *prior, last = parts
    assert last.startswith("Assistant:"), "last turn must be an assistant response"
    prompt = "\n\n".join(prior).strip()
    response = last[len("Assistant:"):].strip()
    return prompt, response


def from_ultrafeedback() -> Iterator[Pair]:
    ds = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train")
    for i, row in enumerate(ds):
        margin = float(row["chosen-rating"]) - float(row["rejected-rating"])
        if margin < 1.0:                                   # per plan
            continue
        yield Pair(
            id=sha1(row["prompt"], row["chosen"][-1]["content"],
                    row["rejected"][-1]["content"], "ufb"),
            source="ultrafeedback-binarized",
            source_row_id=f"row:{i}",
            license="MIT",
            preference_dimension="helpful",
            prompt=row["prompt"],
            chosen=row["chosen"][-1]["content"],
            rejected=row["rejected"][-1]["content"],
            chosen_rank_meta={"score": float(row["chosen-rating"])},
            rejected_rank_meta={"score": float(row["rejected-rating"])},
            agreement={"kind": "pointwise_binarized", "value": margin, "n_raters": 1},
            length_tokens={},
            safety_flags=[],
            decontam={"mod_102_ex_02": "pending"},
        )


def from_oasst() -> Iterator[Pair]:
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    # Build a {parent_id: [children sorted by rank]} map, then emit C(n, 2).
    by_parent: dict[str, list] = {}
    for row in ds:
        if row["role"] != "assistant" or row["rank"] is None:
            continue
        by_parent.setdefault(row["parent_id"], []).append(row)
    for parent_id, siblings in by_parent.items():
        siblings.sort(key=lambda r: r["rank"])
        if len(siblings) < 2:
            continue
        # Reconstruct prompt from the thread (walk up via parent_id). Elided.
        prompt = _oasst_rebuild_prompt(ds, parent_id)
        for a, b in itertools.combinations(siblings, 2):
            if (b["rank"] - a["rank"]) < 2:                 # drop near-tie
                continue
            yield Pair(
                id=sha1(prompt, a["text"], b["text"], "oasst2"),
                source="oasst2",
                source_row_id=f"{a['message_id']}|{b['message_id']}",
                license="Apache-2.0",
                preference_dimension="helpful",
                prompt=prompt,
                chosen=a["text"],
                rejected=b["text"],
                chosen_rank_meta={"rank": int(a["rank"])},
                rejected_rank_meta={"rank": int(b["rank"])},
                agreement={"kind": "ranking", "value": 1.0,
                           "n_raters": int(a.get("count_votes") or 1)},
                length_tokens={},
                safety_flags=[],
                decontam={"mod_102_ex_02": "pending"},
            )


def _oasst_rebuild_prompt(ds, parent_id: str) -> str:
    # Walk parents up to root; render user/assistant alternation.
    # (Elided; see the full pipeline. The important bit is that we use
    # tokenizer.apply_chat_template so the prompt is exactly what the model
    # will see at training time.)
    ...


# ---- polarity flip for safety refusals -------------------------------------

def _is_safety_refusal(text: str) -> bool:
    return bool(re.search(
        r"\b(i can'?t help with that|i won'?t help|i cannot assist|that would be unsafe)\b",
        text, re.IGNORECASE))


def repolarize_for_harmlessness(pair: Pair) -> Pair | None:
    """If the rejected is actually an appropriate safety refusal and the chosen
    is unsafe, flip. Only applied on red-team prompts flagged upstream."""
    if _is_safety_refusal(pair.rejected) and not _is_safety_refusal(pair.chosen):
        pair.chosen, pair.rejected = pair.rejected, pair.chosen
        pair.chosen_rank_meta, pair.rejected_rank_meta = (
            pair.rejected_rank_meta, pair.chosen_rank_meta)
        pair.preference_dimension = "harmless"
        pair.safety_flags.append("repolarized_for_harmlessness")
    return pair


# ---- driver -----------------------------------------------------------------

def main() -> None:
    lsh = MinHashLSH(threshold=0.9, num_perm=64)
    drop_log: dict[str, list[str]] = {}
    kept: list[Pair] = []

    sources = itertools.chain(
        from_hh("train", "helpful-base", "helpful"),
        from_hh("train", "harmless-base", "harmless"),
        from_ultrafeedback(),
        from_oasst(),
    )
    for raw in sources:
        p = filter_and_yield(raw, lsh, drop_log)
        if p is None: continue
        if p.preference_dimension == "harmless":
            p = repolarize_for_harmlessness(p)
        kept.append(p)

    with (OUT / "pairs.jsonl").open("w") as f:
        for p in kept:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    (DROPS / "drop_log.json").write_text(json.dumps(drop_log, indent=2))


if __name__ == "__main__":
    main()
```

### 2.3 Model-as-judge for the synthetic audit slice

```python
# judge_sample.py — judge ~200 synthetic pairs with two independent prompts.
JUDGE_TEMPLATES = [
    # Template A — direct
    "You are evaluating two assistant responses to the same user prompt.\n"
    "Prompt: {prompt}\nResponse A: {a}\nResponse B: {b}\n"
    "Which response is more helpful, accurate, and complete? Answer with "
    "exactly one of: A, B, TIE. Then a one-line reason.",
    # Template B — rubric-anchored
    "Compare two responses on {prompt!r}.\n"
    "Score each on correctness (0-5), completeness (0-5), format (0-5).\n"
    "Then say which has the higher TOTAL score. Output JSON "
    "{{'a_total': int, 'b_total': int, 'winner': 'A'|'B'|'TIE'}}.",
]

def judge_pair(prompt, a, b, client, model="claude-sonnet-4-6") -> str:
    winners = []
    for tmpl in JUDGE_TEMPLATES:
        # Randomize A/B order per template to counter position bias.
        order = RNG.choice([("a", "b"), ("b", "a")])
        ra, rb = (a, b) if order == ("a", "b") else (b, a)
        out = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user",
                       "content": tmpl.format(prompt=prompt, a=ra, b=rb)}],
        ).content[0].text
        w = _parse_winner(out)                     # 'A', 'B', or 'TIE'
        winners.append("a" if (w == "A" and order[0] == "a")
                       or (w == "B" and order[1] == "a") else
                       ("b" if w != "TIE" else "tie"))
    if "tie" in winners or winners[0] != winners[1]:
        return "drop"
    return winners[0]                                # 'a' or 'b'
```

Keep only pairs where both templates and both orderings agree on the same winner; this one hygiene step eliminates the bulk of position bias and template-sensitivity noise that otherwise seeps into the reward model.

### 2.4 `reliability_report.md` — target shape

```markdown
# Preference-Pair Reliability Report — v1

## Pool composition (after all filters)

| Source | Rows in | Rows kept | Keep rate | Dim-helpful | Dim-harmless |
|---|---:|---:|---:|---:|---:|
| hh-rlhf:helpful-base | 42,537 | 1,864 | 4.4% | 1,864 | 0 |
| hh-rlhf:harmless-base | 42,537 | 1,221 | 2.9% | 0 | 1,221 |
| ultrafeedback-binarized | 61,135 | 1,502 | 2.5% | 1,502 | 0 |
| oasst2 (C(n,2), delta≥2) | ~18,000 | 487 | 2.7% | 487 | 0 |
| synthetic (audit) | 500 | 198 | 39.6% | 198 | 0 |
| **Totals** | — | **5,272** | — | **4,051** | **1,221** |

(Illustrative; real keep-rates move with the length cap, the language filter
threshold, and the near-dup threshold. Keep rates on hh-rlhf are low because
hh-rlhf's native pairs are long dialogs; length caps bite.)

## Inter-annotator / inter-judge agreement

| Slice | Agreement metric | Value | Interpretation |
|---|---|---:|---|
| hh-rlhf helpful-base | single rater, no re-rate | n/a | Trust-weight = 0.7 (literature-known noise rate) |
| hh-rlhf harmless-base | single rater | n/a | Trust-weight = 0.8 (narrow policy) |
| UFB binarized (margin ≥ 1.0) | pointwise score margin | mean 2.4, median 2.0 | Larger margins → higher reliability; drop margin < 1.0 |
| OASST ranked (Δrank ≥ 2) | rank distance | mean 2.6, median 2.0 | Small-Δ pairs are the noisiest; drop Δ < 2 |
| Synthetic audit | 2-template agreement | 198/500 = 39.6% kept | 60% of model-judge pairs disagreed between templates — do NOT skip this check |

### Interpretation
- Human-labeled slices (HH, OASST) should set the reward-model ceiling. UFB
  is a cheaper "fill" source but its `chosen-rating - rejected-rating` margin
  is a *confidence* signal — pass it through to the trainer as a sample weight.
- Model-judge agreement at 40% is what you should expect without the
  two-template guard. Teams that skip this guard ship RMs that have learned
  position bias instead of helpfulness.

## Known biases we did not fully correct

- **Length bias.** UFB and GPT-4-judged pairs favor longer responses. We did
  not de-bias length here; reward model (exercise-02) must track length-vs-reward
  correlation on the held-out set as part of its acceptance.
- **Formality bias.** OASST rankings prefer polished prose. If the target
  policy's tone is terse, the RM may mis-rank concise answers.
- **Red-team coverage.** Harmlessness pool comes from a single source
  (hh-rlhf harmless-base, 2022 red-team taxonomy). Known gaps: prompt
  injection, tool-use jailbreaks, multi-turn jailbreaks. These are flagged
  in the data card and handed to mod-106.

## Decontamination
- Overlap against `mod-102 exercise-02` SFT eval sets: 0 prompts (MinHash-LSH
  at Jaccard 0.8).
- Overlap against in-house eval set `rlhf-eval-v1` (500 prompts, kept
  private): 0 (hard match on prompt sha1).

## Splits
Train / val / test = 85 / 10 / 5. Stratified by
`(source, preference_dimension)`. Fixed seed = 0.

## Final shapes
- `pairs.jsonl` : 5,272 rows.
- `pairs_train.jsonl` / `pairs_val.jsonl` / `pairs_test.jsonl` :
  4,481 / 527 / 264.
```

### 2.5 `data_card.md` — minimal skeleton

```markdown
# Preference-Pair Dataset Card — v1

- Name: `prefs-v1`
- Pipeline version: v1 (commit <sha>)
- Rows: 5,272 pairs (4,481 train / 527 val / 264 test)
- Dimensions trained: helpful (4,051), harmless (1,221)
- Sources: Anthropic/hh-rlhf (MIT), argilla/ultrafeedback-binarized (MIT),
  OpenAssistant/oasst2 (Apache-2.0), in-house synthetic (derivative).
- Rendering tokenizer: meta-llama/Meta-Llama-3.1-8B-Instruct, chat template.
- Known biases: length ↑ → reward ↑ correlation on UFB slice; red-team
  taxonomy limited to 2022 harm categories.
- Not suitable for: direct multilingual training, tool-use preference,
  any eval set used upstream in mod-102.
- Reproducibility: `python pairs_pipeline.py --seed 0`.
- Governance: NIST AI RMF Map / Measure (data provenance, bias measurement)
  and Govern (documented intended use). See `reliability_report.md`.
```

### 2.6 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Single `preference_dimension` per row | Mixing helpful + harmless into one RM yields a model that trades them off arbitrarily | One combined head | Either need separate RMs or a multi-objective loss (mod-106 territory) |
| Keep `(chosen, rejected)` not `(prompt, score)` | Pairwise is what Bradley–Terry RMs and DPO/SimPO actually consume [InstructGPT22][DPO23] | Pointwise scores only | Pointwise requires calibration; pairs are directly usable |
| `rank Δ ≥ 2` for OASST C(n,2) | Adjacent-rank pairs are near-ties; they inject noise, not signal | Keep all C(n,2) pairs | More rows but worse RM calibration |
| UFB `margin ≥ 1.0` filter | Pointwise-binarized pairs with tiny margins are judge-noise | No margin filter | ~2× more rows, measurably worse RM on val |
| Two-template judge agreement for synthetic | Position / template bias is large on single-judge LLM labels | Single-template judge | Easy, but reward model learns the bias |
| Repolarize safety refusals | The refusal pattern is the best harmlessness signal in open corpora | Drop refusals | Underpowered harmlessness pool |
| Trust-weight per source (in RM loss) | Not all sources are equally reliable; margin and voter count are confidence signals | Uniform weight | RM fits noisy sources too hard |
| Stratified train/val/test by source × dimension | Avoids leakage of a single source dominating a split | Random split | Split variance across sources |

## 3. Validation steps

1. **Deterministic rebuild.** Running `pairs_pipeline.py --seed 0` on the same raw input twice must produce byte-identical `pairs.jsonl`. If not, something downstream is non-deterministic (hash of a `set()` ordering, un-seeded `random`, etc.).
2. **Decontam against SFT eval.** For every prompt in the mod-102 eval set, MinHash-LSH query against `pairs.jsonl` must return zero hits at Jaccard ≥ 0.8. If a hit shows up, the RM would learn to recognize eval prompts. Hard-fail the pipeline.
3. **Position-bias spot check.** For 50 random pairs, swap `chosen ↔ rejected` and send the pair through the synthetic judge. At least 48/50 must flip their verdict. If more than 2 do not flip, the judge template has a positional bias and the pipeline must retry with `judge_pair` randomized-order.
4. **Trivial-pair audit.** Grep `pairs.jsonl` for rows where `chosen == rejected` or `token_overlap(chosen, rejected) ≥ 0.95`. Must be zero in `pairs.jsonl`; non-zero in `drops/`.
5. **Length-bias readout.** Compute `corr(len(chosen), 1) − corr(len(rejected), 1)` on the train split: `mean(len(chosen) − len(rejected))` and its sign per source. Record the number in the data card; this is what exercise-02's RM is going to learn whether we want it to or not.
6. **Dimension purity.** For 100 random `harmless` rows, confirm the pair elicitation was a safety comparison. If helpful-pool rows leak in, the harmlessness RM learns the wrong objective.
7. **License / provenance completeness.** Every `pairs.jsonl` row has non-null `source`, `source_row_id`, and `license`. `jq 'select(.license == null or .source_row_id == null)' pairs.jsonl` returns nothing.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Plan before code** | No written plan; sources chosen ad hoc | Plan present, missing dimension definition or elicitation semantics | Plan is explicit about sources, dimensions, elicitation, agreement, and licensing before any row is produced |
| **Filter stack** | Only a length filter, or no near-dup check | Length + language + PII only; no near-dup / no decontam | Structural → language → length → PII → near-dup → decontam → trivial-pair, all logged to `drops/` |
| **Preference provenance per row** | Bare `{prompt, chosen, rejected}` with no source | Source + license only | Full schema: source, source_row_id, license, dim, rank meta, agreement, lengths, decontam |
| **Reliability measurement** | No agreement number anywhere | Agreement reported as a single global metric | Agreement broken down by source, with explicit margin/rank-distance filters applied, and a known-bias section |
| **Reproducibility & splits** | Single file, no splits or seed | Splits present but not stratified or not seeded | Fixed seed, stratified by source × dimension, deterministic rebuild, decontam signed off |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Shipping `{prompt, chosen, rejected}` only. Without `source`, `license`, and `preference_dimension`, there is no way to debug which pool is driving the RM.
- Mixing helpful and harmless pairs into one file and training a single RM on both. The RM learns a scalar that is neither helpfulness nor harmlessness, and it usually resolves ambiguity by preferring length.
- Using C(n, 2) pairs from a ranked list without a `Δrank ≥ 2` filter. Near-tie pairs dominate the pool and poison training.
- Using a GPT-4 / Claude "judge" with a single prompt template. Position and template biases bake into every pair; downstream optimizers happily exploit them.
- Dropping safety refusals because they "look wrong". Those pairs — after polarity flip — are the richest harmlessness signal in open corpora.
- Skipping decontamination against the SFT eval set from mod-102. The reward model then rewards policies that happen to answer eval prompts well, and PPO over-optimizes to the eval distribution.
- Reporting agreement as a single global number. The useful breakdown is per source; the global number hides a 95%-agreement source averaged with a 40%-agreement source.
- Letting the length distribution of `chosen` and `rejected` diverge silently. This is the single most common "bias" the resulting RM learns.
- Non-deterministic pipelines. If the same input yields different pairs on reruns, the RM / DPO run is not reproducible, and no audit is possible.
- Treating "preference pairs" as a one-shot artifact. In practice this corpus needs versioning (`v1`, `v2`...) because filters evolve and sources are added; without a version field the RM and policy become un-pinnable to a specific data snapshot.

## 6. References

- [HH22] Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback," 2022. <https://arxiv.org/abs/2204.05862>
- [InstructGPT22] Ouyang et al., "Training language models to follow instructions with human feedback," 2022. <https://arxiv.org/abs/2203.02155>
- [DPO23] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023. <https://arxiv.org/abs/2305.18290>
- [UFB23] Cui et al., "UltraFeedback: Boosting Language Models with High-quality Feedback," 2023. <https://arxiv.org/abs/2310.01377>
- [OASST] Köpf et al., "OpenAssistant Conversations – Democratizing Large Language Model Alignment," 2023. <https://arxiv.org/abs/2304.07327>; dataset: <https://huggingface.co/datasets/OpenAssistant/oasst2>
- `Anthropic/hh-rlhf` dataset. <https://huggingface.co/datasets/Anthropic/hh-rlhf>
- `argilla/ultrafeedback-binarized-preferences`. <https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences>
- `datasketch` MinHash / MinHashLSH. <https://github.com/ekzhu/datasketch>
- Microsoft Presidio (PII detection). <https://microsoft.github.io/presidio/>
- fastText `lid.176` language identification. <https://fasttext.cc/docs/en/language-identification.html>
- OWASP Machine Learning Security Top 10 (ML-02 Data Poisoning, ML-06 AI Supply Chain Vulnerabilities — preference-corpus provenance is a supply-chain surface). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — data-collection and preference-corpus tactics (Resource Development / Initial Access). <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Govern/Map/Measure/Manage; dataset documentation and bias measurement are Measure/Manage obligations). <https://www.nist.gov/itl/ai-risk-management-framework>
