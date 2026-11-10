# mod-102 · exercise-02 — Eval-Set Decontamination — Solution

## 1. Solution overview

Decontamination is the only answer to "did you train on the test set?" that a model card can defend. The exercise walks a full scan against the four canonical reference evals — MMLU [MMLU], HumanEval [HE], GSM8K [GSM], MT-Bench [MTB] — using the recipe published in appendix C of the GPT-3 paper [GPT3] (word-level n-gram overlap) plus an embedding-similarity backstop for the two short-item evals. The deliverable is a report, side-by-side flag examples, and the `decontam` field filled in on every surviving training row.

Three ideas carry the exercise:

- **Two thresholds, both reported.** A strict scan (n=13) that approximates the Llama / GPT-3 recipe and a permissive scan (n=8 with a 1%-of-grams rule) that catches partial leaks. Reporting both makes the FP/precision trade-off explicit.
- **Embedding backstop where n-grams fail.** MT-Bench prompts are short and distinctive; GSM8K problems are often paraphrased in instruction data. N-grams miss paraphrases; cosine similarity ≥ 0.9 catches them.
- **Hand-labeled false-positive rate.** A threshold with no measured FP rate is folklore. The hand-label step turns "0.6% leak" into a defensible number.

## 2. Worked implementation

### 2.1 Eval loaders (Part A)

```python
# eval_corpora.py
from datasets import load_dataset

def mmlu_items():
    ds = load_dataset("cais/mmlu", "all", split="test")
    for i, ex in enumerate(ds):
        # Decision: emit both stem-only (strict) and stem+choices (permissive)
        stem = ex["question"].strip()
        choices = " ".join(ex["choices"])
        yield {"eval":"mmlu","item_id":f"mmlu-{i}","variant":"stem","prompt":stem}
        yield {"eval":"mmlu","item_id":f"mmlu-{i}","variant":"stem+choices",
               "prompt": f"{stem}\n{choices}"}

def humaneval_items():
    ds = load_dataset("openai_humaneval", split="test")
    for ex in ds:
        yield {"eval":"humaneval","item_id":ex["task_id"],
               "variant":"signature+docstring","prompt": ex["prompt"]}
        yield {"eval":"humaneval","item_id":ex["task_id"],
               "variant":"canonical_solution","prompt": ex["canonical_solution"]}

def gsm8k_items():
    ds = load_dataset("gsm8k", "main", split="test")
    for i, ex in enumerate(ds):
        yield {"eval":"gsm8k","item_id":f"gsm8k-{i}",
               "variant":"problem","prompt": ex["question"]}
        yield {"eval":"gsm8k","item_id":f"gsm8k-{i}",
               "variant":"solution","prompt": ex["answer"]}

def mtbench_items():
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    for ex in ds:
        yield {"eval":"mtbench","item_id":str(ex["prompt_id"]),
               "variant":"turn1","prompt": ex["prompt"][0]}
```

Decisions recorded alongside the code:

- **MMLU**: scan both variants separately. Stem-only is the strict pass; stem+choices is permissive because synthetic instruction data often strips MC choices.
- **HumanEval**: scan signatures/docstrings *and* canonical solutions. Solution leaks matter as much as prompt leaks.
- **GSM8K**: problem text and the step-by-step `answer` field are separate corpora. The answer leak teaches the model the chain-of-thought.
- **MT-Bench**: first-turn only. 80 items; any 13-gram match is almost certainly a real leak.

### 2.2 Normalizer and n-gram scanners (Part B)

```python
# ngram_scan.py
import re, unicodedata
from collections import defaultdict
from typing import Iterable

_WS = re.compile(r"\s+")

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    s = _WS.sub(" ", s).strip()
    return s

def shingles(text: str, n: int) -> list[tuple[str, ...]]:
    toks = normalize(text).split()
    if len(toks) < n: return []
    return [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]

def build_index(eval_items, n: int) -> dict[tuple, list[tuple[str,str]]]:
    idx: dict[tuple, list[tuple[str,str]]] = defaultdict(list)
    for item in eval_items:
        for g in shingles(item["prompt"], n):
            idx[g].append((item["eval"], item["item_id"]))
    return idx

def scan_strict(row_text: str, idx: dict, n: int = 13):
    hits = defaultdict(int)         # item_id -> matched gram count
    for g in shingles(row_text, n):
        for ev, item_id in idx.get(g, ()):
            hits[(ev, item_id)] += 1
    if not hits: return []
    # Return top-3 by match count
    return sorted(hits.items(), key=lambda kv: -kv[1])[:3]

def scan_permissive(row_text: str, idx: dict, n: int = 8, ratio: float = 0.01):
    grams = shingles(row_text, n)
    if not grams: return []
    hit_items: dict[tuple[str,str], int] = defaultdict(int)
    for g in grams:
        for ev, item_id in idx.get(g, ()):
            hit_items[(ev, item_id)] += 1
    total = len(grams)
    flagged = {k:v for k,v in hit_items.items() if v / total >= ratio}
    return sorted(flagged.items(), key=lambda kv: -kv[1])[:3]
```

Why these parameters:

- **n=13** follows the GPT-3 recipe [GPT3 Appendix C]; Llama 2 uses `n=10` tokens on their tokenizer and reports it in the model card [Llama2]. The exercise uses word n-grams (tokenizer-agnostic and simpler).
- **n=8, ≥1% of a row's grams** is a conservative-catch that trips on partial lifts without a one-gram-match → false-positive explosion.

### 2.3 Embedding backstop for MT-Bench + GSM8K (Part C)

```python
# embed_scan.py
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_faiss(eval_items):
    texts, ids = [], []
    for item in eval_items:
        texts.append(item["prompt"])
        ids.append((item["eval"], item["item_id"]))
    emb = MODEL.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    return index, ids

def embed_flag(user_prompt: str, index, ids, threshold: float = 0.9):
    q = MODEL.encode([user_prompt], normalize_embeddings=True,
                     convert_to_numpy=True).astype("float32")
    sims, nn = index.search(q, 1)
    s = float(sims[0,0]); j = int(nn[0,0])
    if s >= threshold:
        return {"eval": ids[j][0], "item_id": ids[j][1], "cosine": s}
    return None
```

The embedder choice is reported in the artifact. `all-MiniLM-L6-v2` is fast and cheap; `BAAI/bge-small-en-v1.5` is a stronger alternative and should be documented if used.

### 2.4 Main pipeline

```python
# decontam.py (sketch)
import json, pathlib
from ngram_scan import build_index, scan_strict, scan_permissive, normalize
from embed_scan import build_faiss, embed_flag
from eval_corpora import mmlu_items, humaneval_items, gsm8k_items, mtbench_items

EVALS = {"mmlu":        list(mmlu_items()),
         "humaneval":   list(humaneval_items()),
         "gsm8k":       list(gsm8k_items()),
         "mtbench":     list(mtbench_items())}

# Per-eval ngram indexes at both n values
idx13 = {ev: build_index(items, 13) for ev, items in EVALS.items()}
idx8  = {ev: build_index(items,  8) for ev, items in EVALS.items()}

# Embedding backstop only for the short evals
embed_targets = {ev: build_faiss(EVALS[ev]) for ev in ("mtbench","gsm8k")}

def row_text(row) -> str:
    return normalize(" ".join(m["content"] for m in row["messages"]))

def user_text(row) -> str:
    us = [m["content"] for m in row["messages"] if m["role"] == "user"]
    return normalize(" ".join(us))

def flag(row):
    t = row_text(row); ut = user_text(row)
    out = {ev: {"n13": [], "n8": [], "embed": None} for ev in EVALS}
    for ev in EVALS:
        out[ev]["n13"] = scan_strict(t, idx13[ev], n=13)
        out[ev]["n8"]  = scan_permissive(t, idx8[ev], n=8, ratio=0.01)
    for ev, (idx, ids) in embed_targets.items():
        out[ev]["embed"] = embed_flag(ut, idx, ids, threshold=0.9)
    return out

def worst_label(flags):
    # Priority: embed > n13 > n8 > clean. Removal decided per eval in the report.
    if flags.get("embed"): return "flagged_embed"
    if flags.get("n13"):   return "flagged_ngram_13"
    if flags.get("n8"):    return "flagged_ngram_8"
    return "clean"

def main():
    inp  = pathlib.Path("out/curated.jsonl")
    outp = pathlib.Path("out/curated_decontaminated.jsonl")
    with inp.open() as fin, outp.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            f = flag(row)
            row["decontam"] = {ev: worst_label(f[ev]) for ev in EVALS}
            # Removal decision below (see report); remove rows where any eval is
            # flagged_ngram_13 or flagged_embed on the decided-to-remove evals.
            removable = any(
                row["decontam"][ev] in ("flagged_ngram_13", "flagged_embed")
                for ev in ("mmlu","humaneval","gsm8k","mtbench")
            )
            if removable:
                # record for the report; do not emit
                continue
            fout.write(json.dumps(row) + "\n")
```

### 2.5 FP hand-label protocol (Part D)

For each eval, sample up to 20 flagged rows (or all if fewer). Produce `fp_review.jsonl` with one record per sample:

```jsonc
{
  "eval": "gsm8k",
  "row_id": "oa-7f2a...",
  "matched_item_id": "gsm8k-00412",
  "verdict": "TP" | "FP" | "GREY",
  "reason": "The training row restates the GSM8K problem verbatim (TP)."
}
```

Rule of thumb for calling it:

- **TP** — Row's user prompt would give a model an unfair head start on the eval item.
- **FP** — Overlap is boilerplate math notation, an arithmetic string shared by any two problems, or a trivial eval item whose topic any general training row would cover.
- **GREY** — Same topic/formula, not the same problem. Report separately; do not count as a removal.

### 2.6 `decontam_report.md` — target shape

```markdown
# Decontamination Report

## 1. Eval corpora

| Eval       | Items | Avg words (prompt) | Variants scanned |
|------------|------:|-------------------:|------------------|
| mmlu       | 14,042 (each variant) | 24 / 42 | stem; stem+choices |
| humaneval  | 164 (×2)               | 65 / 48 | signature+docstring; canonical_solution |
| gsm8k      | 1,319 (×2)             | 46 / 110 | problem; solution |
| mtbench    | 80                     | 32 | turn1 |

## 2. Scan parameters

- Normalization: NFC + lowercase + whitespace collapse (`ngram_scan.normalize`).
- Strict n-gram: n=13 word shingles (GPT-3 appendix C recipe, word-level).
- Permissive: n=8 shingles, ≥1% of a row's grams must hit.
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`, cosine ≥ 0.9, applied to user prompt only.

## 3. Flag counts (on curated.jsonl, 41,208 rows)

| Eval       | n=13 strict | n=8 permissive | Embed (≥0.9) |
|------------|------------:|---------------:|-------------:|
| mmlu       |          27 |            612 | — |
| humaneval  |          19 |             84 | — |
| gsm8k      |           6 |            231 | 14 |
| mtbench    |           4 |             17 |  9 |

(Numbers are illustrative; exact counts depend on the curated snapshot.)

## 4. Hand-labeled FP rates (20-item samples)

| Eval       | TP | FP | GREY | FP rate (TP/(TP+FP)) |
|------------|---:|---:|-----:|---------------------:|
| mmlu  n=13 | 18 |  1 |    1 | 5% |
| mmlu  n=8  |  7 | 11 |    2 | 61% |
| humaneval 13 | 17 | 2 |   1 | 11% |
| humaneval 8  | 10 | 8 |   2 | 44% |
| gsm8k n=13 |  5 |  0 |    1 | 0% |
| gsm8k embed| 12 |  3 |    5 | 20% |
| mtbench 13 |  4 |  0 |    0 | 0% |
| mtbench embed| 7 | 2 |    0 | 22% |

## 5. Per-eval decisions

- **MMLU**: Remove on n=13. Keep on n=8 (FP rate too high); annotate only.
- **HumanEval**: Remove on n=13 for both prompt and canonical_solution corpora. Annotate n=8.
- **GSM8K**: Remove on n=13 AND on embed ≥ 0.92 (bumped from 0.9 after FP review).
- **MT-Bench**: Remove on n=13 and on embed ≥ 0.9 (both have 0% FP in sample).

## 6. Aggregate removal

- Rows flagged for removal (union across evals): 61.
- Curated_decontaminated.jsonl row count: 41,147.

## 7. Side-by-side examples

(Ten per eval, with training-row user prompt on the left and matched eval item on the right.)
```

### 2.7 `curated_decontaminated.jsonl` row

```json
{
  "id": "oa-7f2a9d1c04e2",
  "source": "OpenAssistant/oasst1",
  "license": "Apache-2.0",
  "filters_passed": ["structural","language","length","pii","toxicity","response_quality","dedup"],
  "decontam": {
    "mmlu":      "clean",
    "humaneval": "flagged_ngram_8",
    "gsm8k":     "clean",
    "mtbench":   "clean"
  },
  "messages": [ /* ... */ ]
}
```

Row is kept because the only flag is `flagged_ngram_8` on HumanEval, and the report's decision was to annotate (not remove) n=8 flags.

### 2.8 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Two thresholds (n=8, n=13) | Strict matches published recipes [GPT3], [Llama2]; permissive catches partial lifts | Single threshold | Single threshold forces a precision/recall trade-off without exposing it |
| Word n-grams (not token) | Tokenizer-agnostic; reproducible across model families | Tokenizer n-grams | Tokenizer n-grams are closer to what the trainer sees; harder to compare across projects |
| Embed backstop on MT-Bench + GSM8K only | They are short / paraphrasable; MMLU is long enough that n-grams suffice | Embed-scan everything | Embed-scan at scale is expensive and FP-prone on generic topic matches |
| Scan entire conversation (user + assistant) | Leaks happen on either side (response can contain a verbatim HumanEval solution) | Scan user side only | Misses response-side leaks (a model that memorized the solution emits it) |
| Hand-label 20 per eval | Cheapest way to estimate the FP rate honestly | Trust the threshold | A threshold without measured FP rate is a guess |
| Normalize NFC + lowercase + ws collapse | Cheap and reversible; documented | Light / no normalization | Different normalizers across eval and train side silently miss matches |

## 3. Validation steps

1. **Count tie-out.** Rows in `curated.jsonl` − rows removed per the report = rows in `curated_decontaminated.jsonl`.
2. **`decontam` completeness.** `jq 'select(.decontam.mmlu == null or .decontam.humaneval == null or .decontam.gsm8k == null or .decontam.mtbench == null)' curated_decontaminated.jsonl` must be empty. No `not_run` values should remain.
3. **Eval-side normalization parity.** Normalize one MMLU stem with the training-side `normalize()`, split into 13-grams, look up in `idx13["mmlu"]`. Every gram must self-match.
4. **Embedding threshold spot-check.** For MT-Bench, embed each eval prompt and run `embed_flag(eval_prompt, index)` against the index built from the same prompts — every item must return cosine = 1.0 (self-match sanity).
5. **Determinism.** Rerun on the same input + eval snapshot date; flag counts must match exactly.
6. **Hand-label audit.** Spot-check five of the `TP` labels and one of the `GREY` labels per eval with a second reviewer. If either disagrees, revisit the per-eval decision.

## 4. Rubric

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Eval loaders** | Missing an eval or scans unnormalized text | All four evals, single variants, loose normalization | All four evals with both strict + permissive variants where applicable; explicit normalization function |
| **N-gram scanners** | Only one threshold or no top-3 match reporting | Two thresholds but one hardcoded; no per-eval index | Both thresholds, per-eval index, returns top-3 matches |
| **Embed backstop** | Not implemented | Runs but no cosine threshold rationale | Runs on MT-Bench + GSM8K, documented embedder + threshold, self-match sanity check passes |
| **FP hand-label + decisions** | No sample; threshold chosen without measured FP | Sample exists but no per-eval decision | ≥20-item (or exhaustive) sample per eval, documented FP rate, per-eval remove/annotate decision |
| **Report + writeback** | Report missing or `decontam` field not populated | Report exists but counts don't reconcile | ≤10 min read, counts reconcile, every surviving row has a full `decontam` object |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Normalizing training text one way and eval text another — the scan then misses matches. Normalization is a *shared* contract.
- Reporting only the strict threshold. The strict-only report hides partial leaks; the permissive-only report drowns in FPs. Report both.
- Scanning only the user side. HumanEval `canonical_solution` leaks live on the response side.
- Choosing a cosine threshold by vibes. `0.9` is a reasonable starting point; justify what you picked with a sampled FP rate.
- Forgetting the HumanEval canonical-solution corpus. A model that echoes the solution verbatim passes HumanEval without ever having learned to code.
- Counting GREY rows as removals. GREY is a reporting bucket, not a decision — it exists so you can explain why the number did not move.
- Leaving `decontam: {"mmlu": "not_run"}` after this exercise. The field is now the honest record; `not_run` means you did not run it.
- Using a tokenizer-dependent n-gram length and then comparing to someone else's scan. Lengths across tokenizers are not comparable.

## 6. References

- [GPT3] Brown et al., "Language Models are Few-Shot Learners," NeurIPS 2020. Appendix C details the decontamination recipe. <https://arxiv.org/abs/2005.14165>
- [Llama2] Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," 2023. Model card documents n-gram decontam. <https://arxiv.org/abs/2307.09288>
- [Llama3] Dubey et al., "The Llama 3 Herd of Models," 2024. <https://arxiv.org/abs/2407.21783>
- [MMLU] Hendrycks et al., "Measuring Massive Multitask Language Understanding," ICLR 2021. <https://arxiv.org/abs/2009.03300>
- [HE] Chen et al., "Evaluating Large Language Models Trained on Code," 2021. <https://arxiv.org/abs/2107.03374>
- [GSM] Cobbe et al., "Training Verifiers to Solve Math Word Problems," 2021. <https://arxiv.org/abs/2110.14168>
- [MTB] Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 2023. <https://arxiv.org/abs/2306.05685>
- OWASP Machine Learning Security Top 10 — ML-02 Data Poisoning / ML-08 Model Skewing context for test-set contamination. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — ML supply chain / training-data provenance. <https://atlas.mitre.org/>
- NIST AI RMF — Measure/Manage; dataset documentation and model-card obligations. <https://www.nist.gov/itl/ai-risk-management-framework>
- `sentence-transformers/all-MiniLM-L6-v2`. <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- FAISS. <https://github.com/facebookresearch/faiss>
