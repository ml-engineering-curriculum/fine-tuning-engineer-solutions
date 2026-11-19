# mod-108 · exercise-04 — Contamination Detection and Mitigation — Solution

## 1. Solution overview

The deliverable is a **contamination audit report** for a fine-tune run: evidence that the eval set and the training set do not overlap at the n-gram / embedding / behavioral level, plus a written mitigation if overlap is found. Contamination is the single highest-leverage way to inflate a benchmark result; the exercise forces a defensible answer to "could my eval numbers be explained by test-set leakage?"

Four ideas carry the exercise:

- **Contamination is layered; a single detector is not enough.** Exact-match hash catches copies; 13-gram overlap (the LLaMA-2-era standard) catches near-duplicates; embedding-similarity catches paraphrases; behavioral probes catch memorization that none of the above detect. The audit runs all four and stacks their verdicts.
- **The policy is "decontaminate, then re-measure."** Finding leakage is not the end; the mitigation is to remove the contaminated subset from the eval (or from training, whichever is cheaper to fix) and recompute the headline metrics on the decontaminated slice. The report must present the before-and-after.
- **Contamination can be *intentional* (benchmark memorization) or *incidental* (web-scraped training data overlaps a public benchmark).** The investigation records which, because the mitigation differs: intentional contamination is a data-pipeline bug, incidental is a dataset-curation reality that is addressed with held-out splits (exercise 02) and public-benchmark substitution.
- **The exercise's hardest fact is that *not all contamination is detectable*.** A model trained on paraphrased benchmark text may evade n-gram checks; a model fine-tuned on evaluator rationales may evade all four detectors. The report must name the limits of the audit, not claim a clean bill of health.

The worked example audits a fine-tune on a customer-support corpus against the held-out eval from exercise 02 and the public benchmarks from exercise 01.

## 2. Worked implementation

### 2.1 Four layers of detection

| Layer | What it catches | Signal | False-positive risk |
|---|---|---|---|
| **1. Exact-hash** | Identical examples copied between splits | SHA-256 collision on normalized text | Very low (none in practice) |
| **2. N-gram overlap** (13-gram, LLaMA-style) | Near-duplicates, templated examples, boilerplate reuse | Fraction of eval 13-grams present in training | Templated sentences ("the answer is C") cause FPs; detect and allow-list |
| **3. Embedding similarity** | Paraphrases and translations | Max cosine similarity between eval example embeddings and the training corpus's nearest neighbours | High — topical overlap looks like contamination; threshold must be tuned on a known-clean subset |
| **4. Behavioral probes** | Memorized completions | Model's loss on eval examples is anomalously low; or model can complete held-out continuations verbatim | Low but requires access to the model, not just data |

Layers 1–2 are cheap and should run on every (training, eval) pair before any model trains. Layer 3 is medium-cost and runs before every release. Layer 4 is expensive and runs on the release candidate.

### 2.2 Layer 1 — exact-hash detector

```python
# contamination/exact_hash.py
from __future__ import annotations
import hashlib, json, pathlib, re

_WS = re.compile(r"\s+")

def normalize(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())

def _hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()

def build_hash_index(corpus_jsonl: pathlib.Path, field: str) -> set[str]:
    return {_hash(d[field]) for d in _read_jsonl(corpus_jsonl)}

def find_exact_overlaps(train_idx: set[str], eval_jsonl: pathlib.Path, field: str) -> list[dict]:
    hits = []
    for d in _read_jsonl(eval_jsonl):
        if _hash(d[field]) in train_idx:
            hits.append({"doc_id": d["doc_id"], "field": field, "layer": "exact_hash"})
    return hits
```

For a transcript-summarization task, hash both the input transcript *and* the reference summary; a leak on the summary is far worse than a leak on the input.

### 2.3 Layer 2 — 13-gram overlap detector

The 13-gram contamination test is the canonical heuristic used by the LLaMA, PaLM, and GPT-3 teams. Any eval example whose set of 13-grams overlaps the training corpus by more than a threshold is flagged.

```python
# contamination/ngram_overlap.py
from __future__ import annotations
import collections, json, pathlib, re

_WORD = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())

def ngrams(tokens: list[str], n: int = 13) -> set[tuple[str, ...]]:
    if len(tokens) < n: return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

def build_ngram_bloom(train_jsonl: pathlib.Path, field: str, n: int = 13,
                      capacity: int = 500_000_000, error_rate: float = 1e-4):
    """A Bloom filter is the production approach for billions of n-grams;
    for a classroom-scale corpus a plain set() works."""
    from pybloom_live import ScalableBloomFilter
    bloom = ScalableBloomFilter(initial_capacity=capacity, error_rate=error_rate)
    for d in _read_jsonl(train_jsonl):
        for g in ngrams(tokenize(d[field]), n):
            bloom.add(" ".join(g))
    return bloom

def eval_ngram_overlap(eval_jsonl: pathlib.Path, bloom, field: str,
                       n: int = 13, threshold: float = 0.20) -> list[dict]:
    """Threshold is the LLaMA-paper choice: an eval example with ≥ 20% of its
    13-grams present in training counts as contaminated. Templated-answer
    examples (e.g. 'The answer is C.') spuriously trigger and are filtered
    out by stopword density; see _allow_templated below."""
    out = []
    for d in _read_jsonl(eval_jsonl):
        grams = ngrams(tokenize(d[field]), n)
        if not grams:
            continue
        hits = sum(1 for g in grams if " ".join(g) in bloom)
        frac = hits / len(grams)
        if frac >= threshold and not _looks_templated(d[field]):
            out.append({"doc_id": d["doc_id"], "field": field, "layer": "ngram_13",
                        "frac_overlap": round(frac, 3), "hits": hits, "grams": len(grams)})
    return out

_TEMPLATED = {"the answer is", "which of the following", "please choose",
              "multiple choice", "option a"}

def _looks_templated(text: str) -> bool:
    t = text.lower()
    return any(s in t for s in _TEMPLATED) and len(t) < 500
```

**Tuning the threshold**: the LLaMA 2 technical report uses an 80% token-overlap definition on the 8-gram window for its "contaminated" bucket and reports smaller buckets at lower thresholds. For an in-house customer-support fine-tune, 20% at n=13 is a reasonable default; tune on a known-clean subset to calibrate the false-positive rate.

### 2.4 Layer 3 — embedding similarity

```python
# contamination/embedding_sim.py
from __future__ import annotations
import json, pathlib
import numpy as np

def embed(texts: list[str], model) -> np.ndarray:
    """Use a bi-encoder model (e.g. BAAI/bge-small-en-v1.5). Keep it the same
    across train and eval encodings so cosine is meaningful."""
    return model.encode(texts, normalize_embeddings=True, batch_size=64)

def build_index(train_vecs: np.ndarray):
    """FAISS IVF-Flat index on a few million embeddings is enough for the
    held-out eval in exercise 02; for 1B+ scale go HNSW-Flat on-disk."""
    import faiss
    d = train_vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(train_vecs.astype(np.float32))
    return index

def nearest_neighbors(index, eval_vecs: np.ndarray, k: int = 5):
    sims, idx = index.search(eval_vecs.astype(np.float32), k)
    return sims, idx

def flag_paraphrases(eval_rows: list[dict], train_rows: list[dict],
                     sims, idx, cos_threshold: float = 0.92) -> list[dict]:
    """A cosine similarity ≥ 0.92 on bge-small-en-v1.5 is a strong paraphrase
    signal; tune on a labeled calibration set of 100 paraphrase pairs and 100
    near-topic pairs before trusting the threshold."""
    hits = []
    for i, (row, s_row, k_row) in enumerate(zip(eval_rows, sims, idx)):
        if s_row[0] >= cos_threshold:
            hits.append({
                "doc_id": row["doc_id"],
                "layer": "embedding",
                "max_cosine": round(float(s_row[0]), 3),
                "nearest_train_id": train_rows[int(k_row[0])]["doc_id"],
                "top5": [{"train_id": train_rows[int(k)]["doc_id"], "cos": round(float(c), 3)}
                         for c, k in zip(s_row, k_row)],
            })
    return hits
```

**False positives are common** at this layer: two customer-support transcripts about the same billing issue will look near-duplicate even when one is held out. The audit does not *remove* based on this layer alone; it *flags*, a human reviewer confirms, and only confirmed hits go to decontamination.

### 2.5 Layer 4 — behavioral probes

Contamination can be behavioral without being textual: the model knows the answer but did not see the exact string. Two probes are standard.

**Probe A — anomalous loss.** Compute the model's token-level cross-entropy on each eval example. Contaminated examples typically have lower loss than clean ones of similar structure. The test is a one-sided rank-sum between loss on "suspected clean" (e.g., examples in a brand-new post-cutoff slice) and loss on "suspected contaminated" candidates.

```python
# contamination/behavioral.py
from __future__ import annotations
import json, math, pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def per_example_loss(model_id: str, texts: list[str]) -> list[float]:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    losses = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            out = model(**ids, labels=ids["input_ids"])
            losses.append(float(out.loss))
    return losses

def rank_sum_test(losses_clean: list[float], losses_suspect: list[float]) -> dict:
    """Mann-Whitney U, one-sided: are suspect losses smaller than clean losses?"""
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(losses_suspect, losses_clean, alternative="less")
    return {"U": float(u), "p_value_one_sided": float(p),
            "n_clean": len(losses_clean), "n_suspect": len(losses_suspect),
            "median_clean": float(sorted(losses_clean)[len(losses_clean)//2]),
            "median_suspect": float(sorted(losses_suspect)[len(losses_suspect)//2])}
```

**Probe B — prefix completion.** Give the model the first half of an eval example and ask it to complete the rest; measure the character-level or token-level overlap with the true continuation. The MMLU / MATH contamination literature uses this at 20–40 character prefix windows. Memorization of a benchmark produces continuations that match the next token stream for runs far longer than chance.

```python
def prefix_completion_leakage(model, tokenizer, docs: list[dict],
                              prefix_tokens: int = 50, continue_tokens: int = 100) -> list[dict]:
    hits = []
    for d in docs:
        toks = tokenizer(d["text"]).input_ids
        if len(toks) < prefix_tokens + continue_tokens:
            continue
        prefix = toks[:prefix_tokens]
        truth  = toks[prefix_tokens:prefix_tokens + continue_tokens]
        generated = model.generate(
            torch.tensor([prefix]).cuda(),
            max_new_tokens=continue_tokens, do_sample=False,
        )[0][prefix_tokens:prefix_tokens + continue_tokens].tolist()
        match_len = _longest_common_prefix(truth, generated)
        if match_len >= 30:         # tune; chance for a 30-token verbatim match is ~ 0
            hits.append({"doc_id": d["doc_id"], "layer": "prefix_completion",
                         "match_len_tokens": match_len})
    return hits

def _longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n
```

### 2.6 Audit driver — stack the layers

```python
# contamination/audit.py
from __future__ import annotations
import json, pathlib
from collections import defaultdict

def run_audit(train_jsonl, eval_jsonl, model_id, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Layer 1: exact hash — on both input and reference summary.
    train_hashes_input = build_hash_index(train_jsonl, field="source")
    train_hashes_target = build_hash_index(train_jsonl, field="summary")
    hits_exact = find_exact_overlaps(train_hashes_input, eval_jsonl, "source") + \
                 find_exact_overlaps(train_hashes_target, eval_jsonl, "summary")

    # Layer 2: 13-gram overlap on the source (transcripts), which is the big surface.
    bloom = build_ngram_bloom(train_jsonl, field="source", n=13)
    hits_ngram = eval_ngram_overlap(eval_jsonl, bloom, field="source", n=13, threshold=0.20)

    # Layer 3: embedding similarity on the source.
    # Build bi-encoder embeddings for both; FAISS IVF index.
    hits_embed = _run_embedding_pass(train_jsonl, eval_jsonl, cos_threshold=0.92)

    # Layer 4: behavioral probes against the candidate model.
    hits_behavioral = _run_behavioral_pass(model_id, eval_jsonl)

    # Union the per-doc hits into one record per eval example.
    merged: dict[str, dict] = {}
    for h in hits_exact + hits_ngram + hits_embed + hits_behavioral:
        merged.setdefault(h["doc_id"], {"doc_id": h["doc_id"], "layers": []})
        merged[h["doc_id"]]["layers"].append(h["layer"])

    report = {
        "n_eval": sum(1 for _ in _read_jsonl(eval_jsonl)),
        "n_exact_hits": len(hits_exact),
        "n_ngram_hits": len(hits_ngram),
        "n_embedding_hits": len(hits_embed),
        "n_behavioral_hits": len(hits_behavioral),
        "n_any_layer": len(merged),
        "per_layer_set_sizes": {
            "exact_hash":     len({h["doc_id"] for h in hits_exact}),
            "ngram_13":       len({h["doc_id"] for h in hits_ngram}),
            "embedding":      len({h["doc_id"] for h in hits_embed}),
            "behavioral":     len({h["doc_id"] for h in hits_behavioral}),
        },
    }
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2))
    (out_dir / "flagged.jsonl").write_text(
        "\n".join(json.dumps(v) for v in merged.values()))
    return report, merged
```

### 2.7 Mitigation — decontaminate and re-measure

Two mitigations, in order of preference:

**Mitigation A — remove from the eval set.** If contamination was *incidental* (the training corpus happens to contain near-duplicate production transcripts because production traffic is self-similar), the cheapest fix is to subtract the flagged eval rows and recompute the metrics on the clean subset. The decontaminated eval is the new primary; the original is retained as a diagnostic.

```python
# mitigation/decontaminate_eval.py
import json, pathlib
def decontaminate(eval_jsonl: pathlib.Path, flagged_jsonl: pathlib.Path,
                  out_jsonl: pathlib.Path) -> dict:
    flagged = {d["doc_id"] for d in _read_jsonl(flagged_jsonl)}
    kept, dropped = 0, 0
    with out_jsonl.open("w") as f:
        for d in _read_jsonl(eval_jsonl):
            if d["doc_id"] in flagged:
                dropped += 1
            else:
                f.write(json.dumps(d) + "\n")
                kept += 1
    return {"kept": kept, "dropped": dropped,
            "remaining_fraction": kept / max(1, kept + dropped)}
```

**Mitigation B — remove from the training set and retrain.** If contamination was *intentional* (the fine-tune dataset was built by scraping a benchmark page) or if the eval set is small enough that dropping rows destroys per-stratum coverage, the correct fix is to decontaminate the *training* set and rerun the fine-tune. This is expensive (cost = retrain) but is the only mitigation that leaves the eval at its original size.

**After decontamination, re-run exercise 01 and 02's metrics and present both numbers side-by-side.** The report is the before-after table:

| Metric | Before decontamination | After decontamination (n=1127) | Δ |
|---|---:|---:|---:|
| Mean rubric score | 3.91 | 3.78 | −0.13 |
| Acceptable rate | 72.3% | 68.5% | −3.8% |
| Hallucination rate | 1.4% | 1.9% | +0.5% |
| MMLU (from exercise 01) | 52.6% | — | — |

A drop of 3.8% acceptable-rate is inside the pre-defined 5% regression tolerance (exercise 02); if it exceeded the gate, the release would be blocked pending retraining with the decontaminated training set.

### 2.8 Worked numbers (illustrative)

```json
{
  "n_eval": 1200,
  "n_exact_hits": 7,
  "n_ngram_hits": 61,
  "n_embedding_hits": 94,
  "n_behavioral_hits": 12,
  "n_any_layer": 108,
  "per_layer_set_sizes": {
    "exact_hash": 7,
    "ngram_13": 61,
    "embedding": 94,
    "behavioral": 12
  }
}
```

**Reading the output**:

- 7 exact copies are unambiguous — a data-pipeline bug upstream.
- 61 13-gram hits are a mix of paraphrases and incidental overlaps (templated customer greetings).
- 94 embedding hits is an *upper bound* on paraphrase contamination; human review of a 30-row sample converts this into "28 confirmed near-duplicate, 2 topically similar but distinct." Only confirmed hits go to the decontamination step.
- 12 behavioral-probe hits are the strongest signal: the model completes a 50-token prefix with ≥ 30 tokens of exact match. These examples must be removed; they cannot be measured on without a confounder.
- 108 unique rows across layers. The audit emits a `flagged.jsonl` keyed by `doc_id` that downstream `decontaminate_eval.py` consumes.

### 2.9 Directory layout

```
outputs/exercise-04/
├── contamination/
│   ├── exact_hash.py
│   ├── ngram_overlap.py
│   ├── embedding_sim.py
│   ├── behavioral.py
│   └── audit.py
├── mitigation/
│   └── decontaminate_eval.py
├── results/
│   ├── audit.json
│   ├── flagged.jsonl               # one row per flagged doc_id, with layers list
│   ├── human_review.jsonl          # confirmed hits from the embedding-layer sample
│   ├── decontaminated_eval.jsonl
│   └── before_after.md
├── report.md
└── README.md
```

### 2.10 `report.md` skeleton

```markdown
# Contamination audit — tuned model v1

## Setup
- Training corpus: customer-support SFT dataset v1.2 (143k rows)
- Eval set: held-out eval v1.0 (1,200 rows; exercise 02)
- Detectors: exact SHA-256 hash, 13-gram overlap (threshold 0.20),
  bge-small-en-v1.5 cosine (threshold 0.92), behavioral prefix-completion
  (50-token prefix, 30-token match floor) and loss rank-sum.

## Findings
| Layer | Flagged | Confirmed (human-reviewed sample scaled up) | Notes |
|---|---:|---:|---|
| Exact hash | 7 | 7 | Data-pipeline bug; filtered in v1.3 |
| 13-gram | 61 | 54 | 7 templated-greeting FPs allow-listed |
| Embedding (cos ≥ 0.92) | 94 | 70 (± 10) | 30-row manual sample; scaled up |
| Behavioral | 12 | 12 | Strong signal; all removed |
| Union | 108 | 94 (± 10) | 7.8% of eval |

## Mitigation applied
Dropped 108 rows (union of flagged across layers) from the eval, leaving
1,092. Primary metrics recomputed on the 1,092 slice.

## Before vs. after
(see §2.7 before-after table)

## Limitations
- Embedding layer is noisy; threshold was tuned on a 100-pair paraphrase /
  near-topic calibration set but the 24 unconfirmed hits remain uncertain.
- Behavioral probes cannot detect contamination of tasks the model was not
  trained to generate (e.g. multiple-choice evals where the model predicts a
  single letter — the "continuation" is one token).
- Paraphrase contamination with aggressive rewording (translation,
  back-translation) is not reliably detectable by any current technique;
  the time-based partition in exercise 02 is the primary defense, this audit
  is the backstop.

## Next steps
- Patch the training pipeline to apply layers 1–2 before every SFT run, with
  the eval-set bloom filter preloaded. Hard fail on any exact-hash hit.
- Roll v1.3 of the training corpus (the 108 near-duplicates flagged here
  removed from training too) and compare whether the headline rubric
  score moves on the full 1,200-row eval after retrain.
```

## 3. Validation steps

1. **All four layers were run.** `audit.json` has non-null entries for `exact_hash`, `ngram_13`, `embedding`, `behavioral`. A single-layer audit is not an audit.
2. **The 13-gram threshold is documented.** `ngram_13` hits include the threshold used (e.g., 0.20). The LLaMA-paper default is defensible; an unspecified threshold is not.
3. **Embedding hits are human-reviewed on a sample.** `human_review.jsonl` contains at least 30 confirmed/rejected items from the embedding layer; the flagged union is scaled by the confirmed fraction.
4. **Behavioral probes ran against the actual candidate model.** `prefix_completion_leakage` logged the model ID and the prefix/continuation token counts. A behavioral probe against a different checkpoint does not test contamination of the candidate.
5. **Union across layers, not sum.** The `n_any_layer` count deduplicates by `doc_id`; if a row is flagged by two layers it counts once.
6. **Decontaminated eval is a separate file.** `decontaminated_eval.jsonl` exists alongside the original; the original is preserved for diagnostics. A destructive overwrite is a hard fail.
7. **Before-and-after metrics are computed.** `before_after.md` has headline metrics from both runs; if the acceptable-rate dropped > 5%, the report names which stratum lost the most.
8. **Limitations section lists what the audit cannot see.** A report that says "no contamination" after four layers but does not name paraphrase/back-translation / task-level-memorization as blind spots is overclaiming.
9. **Training-pipeline mitigation is proposed.** The report does not stop at "we dropped 108 rows from eval"; it proposes a pipeline fix (layer 1-2 pre-check) that prevents recurrence.
10. **Performance footprint is reasonable.** `audit.py` finishes in under 30 minutes on 1,200 eval rows against a 150k-row training corpus on a single modern GPU; billion-row corpora require Bloom/FAISS infrastructure and are documented as such.

## 4. Rubric for grading

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Detection coverage** | Fewer than 2 layers | 2–3 layers, any combination | All 4 layers (exact, n-gram, embedding, behavioral); layer-specific thresholds documented and defended |
| **Thresholding and calibration** | Thresholds unstated | Thresholds stated but not calibrated | 13-gram threshold defended (LLaMA / PaLM precedent); embedding cosine threshold tuned on a labeled 100-pair calibration set; behavioral-probe match floor justified relative to chance |
| **Human review of noisy layers** | Not attempted | Spot-check without a sample-size argument | ≥ 30-row sample from the embedding layer, confirmed/rejected, with the union-scaled estimate carried through to the report |
| **Before-vs-after metrics** | Not reported | Reported overall only | Reported overall + per-stratum; regression vs. the pre-defined release gate (exercise 02) explicitly checked |
| **Mitigation and limitations** | Only drops rows, no pipeline fix; no limitations section | Pipeline fix named; limitations shallow | Pipeline fix is specific (e.g. "preload eval-set 13-gram bloom in the SFT trainer's dataset builder, hard fail on any exact-hash hit"); limitations section names paraphrase/translation / task-format gaps |

**Pass threshold:** 7/10, with no single dimension at 0 and detection coverage at ≥ 1.

## 5. Common mistakes

- **Single-layer audits.** Exact-hash alone catches none of the real-world cases; n-gram alone misses paraphrases; embedding alone is too noisy to act on. All four together are the audit.
- **No human review on the embedding layer.** Treating every cos-similarity ≥ 0.92 hit as contamination over-decontaminates the eval and inflates the before-after delta. A 30-row human-labeled sample converts "94 flagged" into "70 ± 10 confirmed."
- **Thresholds without justification.** "Flagged everything with cos > 0.7" is a choice; without a calibration set it is also an admission that the audit cannot defend its false-positive rate.
- **Testing the wrong model.** Behavioral probes run against the base model (not the fine-tuned candidate) measure base-model contamination, which is a different question (and bakes in whatever the base was pretrained on).
- **Dropping rows from eval without recomputing metrics.** The decontaminated eval has to produce decontaminated numbers. Reporting the pre-decontamination 72.3% acceptable rate after the fact is one of the easier audit failures to make.
- **Overwriting the original eval file.** The original is the diagnostic; the decontaminated file is the new primary. Both live on disk; neither is destructive of the other.
- **No limitations section.** Contamination is a game of cat and mouse; no audit finds everything. Writing "no contamination found" after four layers claims a certainty no detector provides.
- **Not updating the training pipeline.** The audit is diagnostic; the mitigation is procedural. If the fine-tune's training-data build didn't gate on the eval hash before, the next SFT run will have the same bug.
- **Confusing "training contains eval row" with "training was ever evaluated against eval row."** Supervision leakage is only one kind of contamination. A training set that contains the eval set's *inputs* without targets is still a leak if the model saw them during pretraining; the audit covers both directions.
- **Running n-gram at too low a value of n.** 4-grams and 8-grams produce false-positive floods on templated text; LLaMA-era work converged on n=13 as the practical sweet spot. Use 13 unless you have a documented reason to vary.

## 6. References

- Touvron et al. — *LLaMA 2: Open Foundation and Fine-Tuned Chat Models*. The paper documents the 8-gram token-overlap contamination test and its 80% threshold, which is the lineage behind the 13-gram convention. <https://arxiv.org/abs/2307.09288>
- Dodge et al. — *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*. Introduces the methodology of auditing training corpora against known evaluation benchmarks. <https://arxiv.org/abs/2104.08758>
- Oren et al. — *Proving Test Set Contamination in Black Box Language Models*. Formalizes behavioral-probe contamination detection; the prefix-completion probe above is in this lineage. <https://arxiv.org/abs/2310.17623>
- Sainz et al. — *NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark*. Survey of contamination definitions and detectors. <https://arxiv.org/abs/2310.18018>
- Carlini et al. — *Quantifying Memorization Across Neural Language Models*. The memorization-probe literature. <https://arxiv.org/abs/2202.07646>
- OWASP Machine Learning Security Top 10 — *ML02 Data Poisoning* (directly adjacent: adversarial contamination of training data) and *ML04 Membership Inference Attack* (the "did this row appear in training?" family of attacks that mirrors the audit's behavioral probes). <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — *ML Supply Chain Compromise* (TTP) and the *Reconnaissance → Discover ML Artifacts* tactic apply directly to the contamination story: an adversary who knows the training corpus overlaps a public benchmark can predict evaluation gaps. <https://atlas.mitre.org/>
- NIST AI Risk Management Framework — *Measure 2.3* (model performance under representative conditions) and *Measure 2.7* (safety — the before-after metric is the measurement discipline). <https://www.nist.gov/itl/ai-risk-management-framework>
