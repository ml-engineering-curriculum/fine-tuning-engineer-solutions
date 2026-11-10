# mod-102 · exercise-01 — SFT Data Quality Audit — Solution

## 1. Solution overview

The deliverable is not a clever filter; it is a defensible paper trail. A reviewer should be able to read five files — `rubric.md`, `pipeline.py`, `curated.jsonl`, `drops/`, `audit_report.md` — and reconstruct every row's path from raw to curated.

Three ideas carry the exercise:

- **Rubric before code.** Write what "good" means before you can be nudged by what the data looks like. The rubric is the contract against which every filter is measured.
- **Filter stack, cheapest first.** Structural → language → length → PII → toxicity → response-quality → dedup. Order is a cost choice and a cascade choice: a row that fails structurally never needs a toxicity score.
- **Metadata envelope per row.** Every surviving row ships with its source, license, filter trail, and a `decontam` placeholder that exercise-02 fills in. Bare `{prompt, response}` rows are a bug.

This walkthrough uses `OpenAssistant/oasst1` as the worked corpus [OASST]. The structure applies unchanged to `HuggingFaceH4/no_robots`, `GAIR/lima`, or `databricks/databricks-dolly-15k`.

## 2. Worked implementation

### 2.1 `rubric.md` (what to write, in one page)

Fill in concrete thresholds; everything a filter checks later should trace back here.

```markdown
# oasst1 EN SFT Rubric

## Well-formed example
- Required fields: `messages: [{"role","content"}...]`, non-empty `content` on every turn.
- Roles in {system?, user, assistant}, alternating user→assistant; conversation ends with assistant.
- Tool / function roles: out-of-scope for this pass; drop.

## Quality dimensions (chapter 1)
- Clarity: user turn is a complete request, not a fragment.
- Correctness: assistant answer is self-consistent; factual content not contradicted mid-response.
- Completeness: assistant does not truncate mid-sentence (last char in `.?!"')}]` or code fence).
- Format faithfulness: if the user asked for JSON/code/bullets, the response matches.
- No teacher artifacts: drop "As an AI language model", "I cannot help with that as a language model",
  "Sure! Here is", leaked `<|system|>` fragments.
- Safety fit: benign refusal of benign request → drop. Appropriate refusal of unsafe request → route
  to `safety_refusals.jsonl` (kept for mod-106, not in the SFT mix).

## Language policy
- English only. Detected via fastText `lid.176` on user AND assistant.
  Both must be `en` with prob ≥ 0.80.
- Mixed language (code-switched) → drop.

## PII policy
- Drop rows with any Presidio hit for {EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, IBAN_CODE,
  IP_ADDRESS} on EITHER side. No redaction on this pass (OASST public corpus is small enough
  that dropping is cheaper than redacting responsibly).

## Length policy
- Base tokenizer: `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- Rendered with `apply_chat_template(..., tokenize=True, add_generation_prompt=False)`.
- Target `max_seq_length = 4096`.
- Floor: 16 tokens (anything shorter is noise).
- Ceiling: 4096 tokens (train-time truncation is a separate choice; here we drop).

## Toxicity
- Classifier: Detoxify `original`.
- Drop assistant response if `toxicity > 0.5` OR `severe_toxicity > 0.1`.
- Drop pair outright; do not keep "benign response to toxic prompt" in this exercise.

## Response quality (heuristic floor)
- Unique 4-gram ratio on assistant response ≥ 0.4 (lower ⇒ repetition).
- No match against the teacher-artifact regex set.
- Drop responses whose only content is a URL, "I don't know", or "Sorry, I can't help".

## Refusals
- Appropriate refusals of safety-relevant prompts → `safety_refusals.jsonl` (out-of-scope).
- Refusals of benign tasks → drop.
```

### 2.2 `pipeline.py` — reference structure

```python
# pipeline.py — reference skeleton (illustrative; cut for brevity, not for correctness)
from __future__ import annotations
import hashlib, json, re, pathlib
from dataclasses import dataclass, asdict
from typing import Callable, Iterable

from datasets import load_dataset
from transformers import AutoTokenizer
import fasttext           # or langdetect
from detoxify import Detoxify
from presidio_analyzer import AnalyzerEngine

OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)
DROPS = OUT / "drops"; DROPS.mkdir(exist_ok=True)

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
LID = fasttext.load_model("lid.176.bin")
TOX = Detoxify("original")
PII = AnalyzerEngine()

TEACHER_ARTIFACTS = re.compile(
    r"(as an ai language model|i'm just an ai|sure!? here(?:'s| is)|i cannot and will not|"
    r"as a language model, i|<\|system\|>|<\|end_of_turn\|>)",
    re.IGNORECASE,
)

@dataclass
class FilterResult:
    kept: bool
    reason: str = ""
    extra: dict | None = None

def row_id(source_id: str) -> str:
    return "oa-" + hashlib.sha1(source_id.encode()).hexdigest()[:12]

# --- individual filters ---------------------------------------------------

def structural(ex) -> FilterResult:
    msgs = ex.get("messages") or []
    if not msgs: return FilterResult(False, "empty")
    if any(not m.get("content","").strip() for m in msgs): return FilterResult(False, "empty_turn")
    roles = [m["role"] for m in msgs]
    if roles[-1] != "assistant": return FilterResult(False, "not_assistant_terminal")
    for a,b in zip(roles, roles[1:]):
        if a == b: return FilterResult(False, "role_repeat")
    return FilterResult(True)

def language(ex) -> FilterResult:
    for m in ex["messages"]:
        lab, prob = LID.predict(m["content"].replace("\n"," "))
        if lab[0] != "__label__en" or prob[0] < 0.80:
            return FilterResult(False, "non_english", {"label": lab[0], "p": float(prob[0])})
    return FilterResult(True)

def length(ex) -> FilterResult:
    ids = TOKENIZER.apply_chat_template(ex["messages"], tokenize=True,
                                        add_generation_prompt=False)
    n = len(ids)
    if n < 16:   return FilterResult(False, "too_short", {"n": n})
    if n > 4096: return FilterResult(False, "too_long",  {"n": n})
    return FilterResult(True, "", {"n_tokens": n})

def pii(ex) -> FilterResult:
    for m in ex["messages"]:
        res = PII.analyze(text=m["content"], language="en",
                          entities=["EMAIL_ADDRESS","PHONE_NUMBER","CREDIT_CARD",
                                    "US_SSN","IBAN_CODE","IP_ADDRESS"])
        if res:
            return FilterResult(False, "pii", {"entities": [r.entity_type for r in res]})
    return FilterResult(True)

def toxicity(ex) -> FilterResult:
    last = ex["messages"][-1]["content"]
    score = TOX.predict(last)
    if score["toxicity"] > 0.5 or score["severe_toxicity"] > 0.1:
        return FilterResult(False, "toxic", {"scores": score})
    return FilterResult(True)

def response_quality(ex) -> FilterResult:
    r = ex["messages"][-1]["content"]
    toks = r.split()
    if len(toks) < 8: return FilterResult(False, "short_response")
    grams = list(zip(toks, toks[1:], toks[2:], toks[3:]))
    if grams:
        uniq = len(set(grams)) / len(grams)
        if uniq < 0.4: return FilterResult(False, "repetitive", {"uniq": uniq})
    if TEACHER_ARTIFACTS.search(r):
        return FilterResult(False, "teacher_artifact")
    return FilterResult(True)

def dedup_key(ex) -> str:
    joined = "\n".join(m["content"].strip().lower() for m in ex["messages"])
    return hashlib.sha256(joined.encode()).hexdigest()

# --- pipeline -------------------------------------------------------------

STAGES: list[tuple[str, Callable]] = [
    ("structural", structural),
    ("language",   language),
    ("length",     length),
    ("pii",        pii),
    ("toxicity",   toxicity),
    ("response_quality", response_quality),
]

def iter_oasst_en_assistant_terminated() -> Iterable[dict]:
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    # Reduce threads to the assistant-preferred linear path and emit each
    # (system?, user, assistant, user, assistant, ...) ending in assistant.
    # Implementation left as an exercise; the scaffold here is schema-stable.
    for ex in ds:
        # placeholder: in practice, build threads from (parent_id, message_tree_id, role, rank).
        yield ex  # pragma: no cover

def run():
    counts = {s: [0,0] for s,_ in STAGES}        # [in, kept]
    counts["dedup"] = [0,0]
    seen_hashes: set[str] = set()
    curated = open(OUT / "curated.jsonl", "w")
    drop_files = {s: open(DROPS / f"{s}.jsonl", "w") for s,_ in STAGES}
    drop_files["dedup"] = open(DROPS / "dedup.jsonl", "w")

    for raw in iter_oasst_en_assistant_terminated():
        row = {"messages": raw["messages"], "source_id": raw["id"]}
        passed = []
        dropped = False
        for name, fn in STAGES:
            counts[name][0] += 1
            r = fn(row)
            if not r.kept:
                drop_files[name].write(json.dumps({"reason": r.reason,
                                                   "extra": r.extra,
                                                   "row": row}) + "\n")
                dropped = True
                break
            counts[name][1] += 1
            passed.append(name)
        if dropped:
            continue

        h = dedup_key(row)
        counts["dedup"][0] += 1
        if h in seen_hashes:
            drop_files["dedup"].write(json.dumps({"reason":"exact_dup","row":row}) + "\n")
            continue
        seen_hashes.add(h)
        counts["dedup"][1] += 1
        passed.append("dedup")

        curated.write(json.dumps({
            "id": row_id(row["source_id"]),
            "source": "OpenAssistant/oasst1",
            "source_id": row["source_id"],
            "license": "Apache-2.0",
            "source_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
            "category": "general_chat",
            "language": "en",
            "generator": None,
            "generator_prompt_hash": None,
            "quality_score": None,
            "filters_passed": passed,
            "decontam": {"mmlu":"not_run","humaneval":"not_run",
                         "gsm8k":"not_run","mtbench":"not_run"},
            "messages": row["messages"],
        }) + "\n")

    for f in [curated, *drop_files.values()]: f.close()
    with open(OUT / "counts.json", "w") as f:
        json.dump(counts, f, indent=2)

if __name__ == "__main__":
    run()
```

Notes on the reference:

- Order is cheapest → most expensive: structural is pure Python, toxicity loads a classifier. A row that fails structurally never runs a Detoxify pass.
- Only Detoxify sees the assistant response (per rubric). Running it on user turns is a different and easy-to-defend choice; just say so in the rubric.
- The dedup key hashes the full normalized conversation — not just the prompt — so near-identical responses with different prompts both survive.
- The license string comes from the source dataset's SPDX, not from an environment variable. `oasst1` is `Apache-2.0` on its HF repo [OASST].

### 2.3 Example `curated.jsonl` row

```json
{
  "id": "oa-7f2a9d1c04e2",
  "source": "OpenAssistant/oasst1",
  "source_id": "d9e3f0c7-...",
  "license": "Apache-2.0",
  "source_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
  "category": "general_chat",
  "language": "en",
  "generator": null,
  "generator_prompt_hash": null,
  "quality_score": null,
  "filters_passed": ["structural","language","length","pii","toxicity","response_quality","dedup"],
  "decontam": {"mmlu":"not_run","humaneval":"not_run","gsm8k":"not_run","mtbench":"not_run"},
  "messages": [
    {"role": "user", "content": "Explain the difference between HTTP and HTTPS in two sentences."},
    {"role": "assistant", "content": "HTTP is the plaintext protocol..."}
  ]
}
```

### 2.4 `audit_report.md` — target shape

```markdown
# oasst1 EN SFT Audit Report

**Input:** 84,437 assistant-terminated English threads (post-thread-linearization).
**Output:** `curated.jsonl`, 41,208 rows.
**Rubric:** rubric.md (committed alongside this report).

## Per-filter drop rates

| Stage              | In     | Kept   | Dropped | Drop rate |
|--------------------|--------|--------|---------|-----------|
| structural         | 84,437 | 83,901 | 536     | 0.6%  |
| language           | 83,901 | 68,412 | 15,489  | 18.5% |
| length             | 68,412 | 66,330 | 2,082   | 3.0%  |
| pii                | 66,330 | 65,117 | 1,213   | 1.8%  |
| toxicity           | 65,117 | 63,994 | 1,123   | 1.7%  |
| response_quality   | 63,994 | 42,640 | 21,354  | 33.4% |
| dedup (exact)      | 42,640 | 41,208 | 1,432   | 3.4%  |

(Counts above are indicative; exact numbers depend on the OASST snapshot date
and the thread-linearization choices. The *shape* — language and response-quality
filters doing most of the work on this corpus — is stable across runs.)

## Drop samples (5 per filter)

For each stage, five dropped rows are listed in `drops/<stage>.jsonl`.
Each sample is labeled `TP` (filter was right), `FP` (filter was wrong),
or `GREY` (defensible either way).

### response_quality — sample 3 of 5 — FP

> Response: "Yes." (6 tokens, dropped as `short_response`.)
> Judgment: FP. The user asked a yes/no question ("Does 2+2 equal 4?") and a
> one-word answer is the correct format. Fix: carve out a yes/no-prompt
> exception on `short_response`, or raise the floor only on open-ended prompts.
> Not fixed in this pass — logged as known issue.

### language — sample 1 of 5 — TP

> User: "Привет, как дела?" / Assistant: "Привет! У меня всё хорошо..."
> Dropped: `non_english` (ru). Correct per rubric.

(Repeat for every stage.)

## Distributions

- Language (post-filter): en 100% (by construction).
- Token-length histogram (Llama 3.1 tokenizer): median 312, p95 1,842, max 4,091.
- Category: single `general_chat` label (source does not provide categories).

## Deviations from rubric

- Added `--short-yesno-bypass` carve-out after the FP review surfaced many single-word
  valid answers. Rubric updated in the same commit.

## Known issues

1. `decontam` is `not_run` across the board — exercise-02 fills it in.
2. Exact dedup only; near-dedup (MinHash) is in the stretch goal.
3. Single-language slice; multilingual variants need a re-run.
4. Toxicity classifier is `detoxify/original` (English, general-purpose). It misses
   domain-specific abuse categories; see HAP or OpenAI Moderation for bigger coverage.
```

### 2.5 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Rubric before code | Prevents post-hoc justification; makes filters testable against a fixed target | Iterate filters and backfill the rubric | Faster start, no auditability |
| Cheap → expensive stage order | Toxicity / tokenizer runs only on rows that already pass structural + language | Any order | Wall-clock cost on large corpora |
| Drop vs. redact PII | Open corpora are small; dropping is simpler and defensible | Redact in place with Presidio | Redaction preserves volume; Presidio false-positives on code-ish content |
| Exact-hash dedup | Zero FP; catches the dominant case on small single-source corpora | MinHash+LSH near-dedup | Near-dedup catches paraphrases; added FPs, more config |
| English-only | Matches the stated target; keeps downstream evals interpretable | Multilingual keep | Multilingual mixes change evaluation + mix-ratio decisions |
| Detoxify on assistant turn only | Rubric treats assistant content as what the model learns to emit; user-side toxicity is addressed by response policy | Also score user | Drops more rows; separate policy choice |
| `apply_chat_template`-based length | The number the trainer actually sees, not characters | Character length | Character counts diverge badly from token counts on code / non-English |

## 3. Validation steps

1. **Rubric–filter traceability.** Open `rubric.md` and `pipeline.py` side by side. Every numeric threshold in the code (0.80 language prob, 16/4096 length bounds, 0.4 n-gram ratio, 0.5 toxicity) must appear in the rubric. If it does not, either the rubric is incomplete or the pipeline added an implicit rule.
2. **Count reconciliation.** `jq -s 'length' curated.jsonl` must equal the final `[kept]` number in `counts.json`. `wc -l drops/*.jsonl | tail -1` plus curated must equal the input count, within rounding from linearization.
3. **Metadata completeness.** `jq 'select(.license==null or .license=="" or .source==null or .id==null)' curated.jsonl` must be empty. No `null`s in required fields.
4. **Drop-sample audit.** Manually read the five drop samples per filter. For each filter, at least one sample must be labeled — not inventing FPs, but looking for them. If every drop is TP, the review was shallow.
5. **Round-trip tokenization.** For 50 random surviving rows, apply the declared tokenizer with `add_generation_prompt=False` and verify `len(input_ids) ≤ 4096`. The length filter must be honest post-templating, not pre-templating.
6. **Determinism.** Rerun on the same input with the same filter versions; counts must match exactly. If they do not, a filter is non-deterministic (e.g., langid with a wrong seed) and must be pinned.

## 4. Rubric for grading the learner's audit

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Rubric quality** | No rubric, or rubric written after the filters | Rubric exists but ambiguous; thresholds not numeric | Explicit thresholds, language/PII/safety policy each addressed, tokenizer + max_seq_length named |
| **Filter coverage** | Missing one of the seven stages or stages out of order | All stages present but some trivially configured | All stages, cheapest-first order, each threshold traceable to the rubric |
| **Metadata envelope** | Bare `{prompt, response}` or missing required fields | Partial envelope, some `null` on required fields | Full envelope per row; license is a real SPDX; `decontam` is `not_run` (not missing) |
| **Drop samples + FP review** | No drop samples, or all drops marked TP without inspection | Drop samples exist but no FP identified | ≥5 samples per filter; at least one filter has a labeled FP with a stated fix |
| **Audit report** | Report missing, or counts don't reconcile with artifacts | Counts tie out but no "known issues" or no deviations recorded | Report ≤5 min read, counts tie out, deviations + known issues both present |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Writing the rubric last so it justifies whatever the filters already did.
- Tokenizing with one tokenizer for the length filter and a different one for training. The filter's job is to predict what the trainer will see.
- Running Detoxify on code snippets — profanity in comments counts; decide whether that is a true positive for your target.
- Hashing only the prompt for dedup — leaves near-duplicate responses with reworded prompts in the set.
- Keeping benign refusals as regular SFT rows. The model learns to refuse the user's benign tasks.
- Dropping `source_url` because "it's obvious" — it isn't once the dataset is six months downstream.
- Collapsing stages into one pass to "save time" — you lose the per-stage drop rates that make the audit defensible.
- Treating thread linearization as a filter. It's a preprocessing choice; its drops do not belong in the filter stack's accounting.

## 6. References

- [OASST] LAION et al., "OpenAssistant Conversations — Democratizing Large Language Model Alignment," 2023. Dataset card: <https://huggingface.co/datasets/OpenAssistant/oasst1>. License: Apache-2.0.
- [LIMA] Zhou et al., "LIMA: Less Is More for Alignment," NeurIPS 2023. <https://arxiv.org/abs/2305.11206>
- [Alpaca] Taori et al., "Stanford Alpaca: An Instruction-following LLaMA Model," 2023. <https://crfm.stanford.edu/2023/03/13/alpaca.html>
- [Dolly] Databricks, "Free Dolly," 2023. Dataset: <https://huggingface.co/datasets/databricks/databricks-dolly-15k>
- [NoRobots] HuggingFace H4, "No Robots." <https://huggingface.co/datasets/HuggingFaceH4/no_robots>
- OWASP Machine Learning Security Top 10 (ML-02 Data Poisoning context for dataset-quality controls). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — tactics/techniques around ML supply chain and data integrity. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework (Govern/Map/Measure/Manage; dataset documentation obligations). <https://www.nist.gov/itl/ai-risk-management-framework>
- Presidio (PII detection). <https://microsoft.github.io/presidio/>
- fastText `lid.176`. <https://fasttext.cc/docs/en/language-identification.html>
- Detoxify. <https://github.com/unitaryai/detoxify>
- SPDX License List. <https://spdx.org/licenses/>
