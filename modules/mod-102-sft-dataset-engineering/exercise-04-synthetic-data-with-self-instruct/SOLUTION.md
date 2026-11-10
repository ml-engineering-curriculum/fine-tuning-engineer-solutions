# mod-102 · exercise-04 — Synthetic Data with Self-Instruct — Solution

## 1. Solution overview

Self-Instruct [SI23] bootstraps a synthetic SFT set from a small hand-authored seed pool by iteratively asking a teacher model to generate new, distinct instructions, generating responses for them, filtering, and recycling survivors back into the pool. Evol-Instruct [EI23] layers *depth* and *breadth* operators on top (add_constraint, deepen, concretize, …) to push the hard tail of the distribution.

The deliverable is small on purpose — 200 surviving pairs — because that is enough to see every failure mode of the loop (teacher drift, refusal bursts, category collapse, near-duplicate explosion) and because it keeps the teacher bill under $5 on hosted providers and under $0 on a local model. The emphasis is on the paperwork: the license decision made *before* generation, the provenance record attached to every row, and the drop-rate numbers the reviewer can independently recompute.

## 2. Worked implementation

### 2.1 `teacher_license.md` (Part A) — worked answer

```markdown
# Teacher-License Decision

## 1. Chosen teacher
- Local vLLM serving `meta-llama/Meta-Llama-3.1-70B-Instruct` (via a lab-managed endpoint).

## 2. License / terms
- Model weights distributed under the Llama 3.1 Community License.
- Outputs: the Community License explicitly permits use of outputs, with named
  restrictions (naming requirement for derivative models — "Llama" prefix;
  commercial-use MAU threshold).
- <!-- pending-citation: on the next research cycle, confirm the current Llama 3.1
  Community License clauses on (a) synthetic-data derivatives used for training
  other models, (b) the 700M MAU commercial threshold, (c) the "Built with Llama"
  naming requirement, with URLs and effective dates -->

## 3. Downstream redistribution
- A model fine-tuned on these outputs is redistributable under the Llama 3.1
  Community License, subject to the naming and MAU conditions above.
- Commercial redistribution is permitted within those limits.
- Attribution: the derived model's model card must disclose that its training
  data includes Llama-3.1-70B-Instruct outputs.

## 4. Obligations the data carries
- Attribution: disclose teacher model and version in every downstream model card.
- Naming: if a derivative model is released, its name must carry the "Llama"
  prefix (community-license requirement).
- No use of outputs to train competing non-Llama foundation models beyond the
  license's permitted purposes. <!-- pending-citation: exact clause reference -->

## 5. Alternatives considered and rejected
- OpenAI GPT-4o: Terms of Use prohibit using outputs to train competing LLMs;
  blocks our intended downstream use. Rejected.
- Anthropic Claude: similar restrictions. Rejected.
  <!-- pending-citation: current OpenAI and Anthropic Terms sections
       governing outputs-for-training, with URLs and effective dates -->
- `nvidia/Nemotron-4-340B-Instruct`: NVIDIA Open Model License explicitly
  permits synthetic-data generation for model training; strong alternative,
  kept as a backup teacher if Llama 3.1 is not available.
  <!-- pending-citation: confirm NOML terms, effective date, URL -->
```

(Any `pending-citation` comments block auto-merge and get resolved on the next research cycle. They are honest placeholders, not excuses.)

### 2.2 `seeds.jsonl` (Part B) — shape + a worked example

20 hand-authored seeds spanning three categories: `reasoning`, `code`, `general_chat`. Example seed:

```json
{"seed_id":"seed-04","category":"reasoning",
 "messages":[
   {"role":"user","content":"Three gears interlock A-B-C. A turns clockwise at 60 rpm. B has twice the teeth of A, C has half the teeth of B. What is C's direction and rpm?"},
   {"role":"assistant","content":"B turns counter-clockwise at 30 rpm (twice the teeth → half the speed, opposite direction). C turns clockwise at 60 rpm (half the teeth of B → twice the speed, opposite direction, which returns to the original direction). Answer: C turns clockwise at 60 rpm."}
 ]}
```

Design constraints that make seeds worth their weight:

- No two seeds are the same category *and* the same difficulty — the pool is for diversity priming, not volume.
- Each seed response is of the length and format you want the model to imitate (short reasoning chains, fenced code with one line of prose).
- Seeds were written by the engineer. If any came from a dataset, this exercise is contaminated at step one.

### 2.3 `pipeline.py` — Self-Instruct loop (Part C)

```python
# pipeline.py
from __future__ import annotations
import hashlib, json, random, pathlib
from dataclasses import dataclass, field
from typing import Iterable

from rouge_score.rouge_scorer import RougeScorer
from datasketch import MinHash

from teacher_client import generate  # your vLLM / OpenAI-compat client
from filters import refusal_detected, code_parses, math_has_boxed_or_answer

SEED = 1337
TARGET = 200
MAX_CALLS = 2000                      # cap teacher calls
TEMPERATURE = 0.7
TEACHER_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

random.seed(SEED)

GEN_PROMPT = (
    "You are an instruction-generation assistant.\n\n"
    "Below are example instructions written by expert humans. "
    "Generate ONE new instruction that:\n"
    "- is on a clearly different topic from every example,\n"
    "- is of comparable or higher difficulty,\n"
    "- has a single clear, verifiable answer,\n"
    "- requires no browsing or files to answer,\n"
    "- is a single self-contained paragraph.\n\n"
    "Return only the instruction text, no preamble.\n\n"
    "Examples:\n{examples}\n\nNew instruction:"
)
RESPONSE_PROMPT = (
    "Answer the following instruction clearly, concisely, and correctly.\n"
    "Instruction:\n{instruction}\n\nAnswer:"
)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

@dataclass
class Row:
    rid: str
    category: str
    instruction: str
    response: str
    seeds: list[str]
    evol: list[dict] = field(default_factory=list)
    filters_passed: list[str] = field(default_factory=list)

def load_seeds() -> list[dict]:
    return [json.loads(l) for l in open("seeds.jsonl")]

def minhash(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for w in text.lower().split():
        mh.update(w.encode())
    return mh

# --- filters --------------------------------------------------------------

ROUGE = RougeScorer(["rougeL"], use_stemmer=True)
DEDUP_ROUGE_THRESHOLD = 0.7

def dedup_hit(new_text: str, pool: list[Row]) -> tuple[bool, Row | None]:
    best: tuple[float, Row | None] = (0.0, None)
    for r in pool:
        f = ROUGE.score(r.instruction, new_text)["rougeL"].fmeasure
        if f > best[0]:
            best = (f, r)
    return best[0] >= DEDUP_ROUGE_THRESHOLD, best[1]

def category_of(instruction: str) -> str:
    s = instruction.lower()
    if any(k in s for k in ("python", "javascript", "write a function",
                            "implement", "pseudocode", "algorithm")):
        return "code"
    if any(k in s for k in ("solve", "proof", "sum", "integer", "prob",
                            "calculate", "theorem")):
        return "reasoning"
    return "general_chat"

# --- Evol operator --------------------------------------------------------

EVOL_PROMPT = (
    "Rewrite the instruction below to add exactly ONE additional constraint "
    "that makes it strictly harder, without changing the task type. "
    "Return only the new instruction.\n\nOriginal:\n{instruction}\n\nRewritten:"
)

def evol_add_constraint(instr: str) -> str:
    return generate(model=TEACHER_MODEL, prompt=EVOL_PROMPT.format(instruction=instr),
                    temperature=TEMPERATURE).strip()

# --- main loop ------------------------------------------------------------

def run():
    seeds = load_seeds()
    pool: list[Row] = []
    raw_log = open("raw_generations.jsonl", "w")
    refusals = open("refusals.jsonl", "w")
    drops = {k: open(f"drops/{k}.jsonl", "w") for k in ("dedup","refusal","format")}
    pathlib.Path("drops").mkdir(exist_ok=True)

    calls = 0
    round_i = 0
    while len(pool) < TARGET and calls < MAX_CALLS:
        round_i += 1
        chosen_seeds = random.sample(seeds, 3)
        chosen_pool  = random.sample(pool, min(3, len(pool)))
        examples = "\n".join(
            f"- {s['messages'][0]['content']}" for s in chosen_seeds + [
                {"messages":[{"content": r.instruction}]} for r in chosen_pool
            ]
        )
        gp = GEN_PROMPT.format(examples=examples)
        instr = generate(model=TEACHER_MODEL, prompt=gp,
                         temperature=TEMPERATURE).strip()
        calls += 1

        # Optional Evol: every 5th round after warmup
        evol_history = []
        if round_i > 20 and round_i % 5 == 0:
            instr2 = evol_add_constraint(instr)
            evol_history = [{"operator":"add_constraint","round":round_i,
                             "prior_text_hash": sha256(instr)}]
            instr = instr2
            calls += 1

        rp = RESPONSE_PROMPT.format(instruction=instr)
        resp = generate(model=TEACHER_MODEL, prompt=rp,
                        temperature=TEMPERATURE).strip()
        calls += 1

        raw_log.write(json.dumps({
            "round": round_i, "instr": instr, "resp": resp,
            "gen_prompt_hash": sha256(gp),
            "evol": evol_history,
            "seed_ids": [s["seed_id"] for s in chosen_seeds],
        }) + "\n")

        cat = category_of(instr)

        # --- filters ---
        dup, near = dedup_hit(instr, pool)
        if dup:
            drops["dedup"].write(json.dumps({
                "rejected": instr, "near_neighbor": near.instruction,
                "rougeL": DEDUP_ROUGE_THRESHOLD,
            }) + "\n")
            continue
        if refusal_detected(resp):
            refusals.write(json.dumps({"instruction": instr, "response": resp}) + "\n")
            continue
        if cat == "code" and not code_parses(resp):
            drops["format"].write(json.dumps({"instr": instr, "resp": resp,
                                               "reason": "ast_parse_failed"}) + "\n")
            continue
        if cat == "reasoning" and not math_has_boxed_or_answer(resp):
            drops["format"].write(json.dumps({"instr": instr, "resp": resp,
                                               "reason": "no_boxed_answer"}) + "\n")
            continue

        rid = f"syn-{len(pool):06d}"
        pool.append(Row(
            rid=rid, category=cat, instruction=instr, response=resp,
            seeds=[s["seed_id"] for s in chosen_seeds],
            evol=evol_history,
            filters_passed=["dedup","refusal_ok","format"],
        ))

    # --- write synthetic.jsonl ---
    with open("synthetic.jsonl", "w") as out:
        for r in pool:
            out.write(json.dumps({
                "id": r.rid,
                "source": "self-instruct-local",
                "license": "Llama-3.1-Community-License (teacher outputs)",
                "category": r.category,
                "generator": TEACHER_MODEL,
                "generator_prompt_hash": sha256(GEN_PROMPT),
                "generator_temperature": TEMPERATURE,
                "generator_seed_ids": r.seeds,
                "evol_history": r.evol,
                "filters_passed": r.filters_passed,
                "messages": [
                    {"role":"user","content": r.instruction},
                    {"role":"assistant","content": r.response},
                ],
            }) + "\n")

if __name__ == "__main__":
    run()
```

### 2.4 `filters.py` — refusal / format checks (Part E)

```python
# filters.py
import ast, re

_REFUSAL = re.compile(
    r"(i can'?t help with that|i cannot (?:help|assist|provide)|"
    r"i'?m (?:not able|unable) to|as an ai language model,? i cannot|"
    r"i must decline|i will not)",
    re.IGNORECASE,
)

def refusal_detected(text: str) -> bool:
    return bool(_REFUSAL.search(text))

_CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

def code_parses(text: str) -> bool:
    m = _CODE_FENCE.search(text)
    if not m: return False
    try:
        ast.parse(m.group(1))
        return True
    except SyntaxError:
        return False

_ANSWER = re.compile(r"(\\boxed\{[^}]+\}|the answer is [-+]?\d|answer:\s*[-+]?\d)",
                     re.IGNORECASE)

def math_has_boxed_or_answer(text: str) -> bool:
    return bool(_ANSWER.search(text))
```

### 2.5 `synthetic.jsonl` — worked row

```json
{
  "id": "syn-000123",
  "source": "self-instruct-local",
  "license": "Llama-3.1-Community-License (teacher outputs)",
  "category": "reasoning",
  "generator": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "generator_prompt_hash": "f3a2c7...e81",
  "generator_temperature": 0.7,
  "generator_seed_ids": ["seed-04","seed-11","seed-17"],
  "evol_history": [{"operator":"add_constraint","round":35,"prior_text_hash":"b19a..."}],
  "filters_passed": ["dedup","refusal_ok","format"],
  "messages": [
    {"role":"user","content":"Three gears interlock A-B-C where A has 20 teeth, B has 40, C has 10. A spins clockwise at 60 rpm. All gears mesh on the same plane. Also assume 5% slip at each mesh. Compute C's direction and rpm, rounded to one decimal place."},
    {"role":"assistant","content":"With no slip, B turns ccw at 30 rpm, C turns cw at 120 rpm. Each mesh loses 5%: 30 × 0.95 = 28.5 rpm for B; 28.5 × 4 × 0.95 = 108.3 rpm for C. Direction: clockwise. Answer: 108.3 rpm."}
  ]
}
```

### 2.6 `synth_report.md` — target shape

```markdown
# Synthetic SFT Mini-Report

## Totals
- Raw teacher generations (instruction + response pairs): 412
- After dedup: 318
- After refusal routing: 304 (14 refusals in refusals.jsonl)
- After format filter: 200
- Final: 200 rows in synthetic.jsonl

## Per-filter drop rate
| Filter | In | Kept | Drop rate |
|---|---:|---:|---:|
| dedup (RougeL ≥ 0.7) | 412 | 318 | 22.8% |
| refusal              | 318 | 304 |  4.4% |
| format (code AST / math answer) | 304 | 200 | 34.2% |

## Category distribution
| Category | Seeds | Final pool | Drift |
|---|---:|---:|---:|
| reasoning | 8 (40%) | 92 (46%) | +6pp |
| code      | 7 (35%) | 61 (30.5%) | -4.5pp |
| general_chat | 5 (25%) | 47 (23.5%) | -1.5pp |

Observation: Evol-Instruct `add_constraint` rounds skew toward reasoning.
Starting round 20, +6pp drift aligns with Evol activation.

## Sampled dropped rows (5 per filter)
(dedup, refusal, format — see drops/ and refusals.jsonl)

## Hand-rated quality on 10 random survivors
| rid | hand rating (1–5) | note |
|---|---:|---|
| syn-000007 | 4 | solid reasoning chain |
| syn-000022 | 3 | correct but overlong |
| syn-000041 | 2 | code runs, but doesn't solve the stated problem |
| ... | ... | ... |
Average: 3.6/5. Percentage ≤ 2: 10%.

## Next iteration
- Tighten dedup threshold to 0.6 to cut near-duplicate reasoning chains.
- Add a judge-model filter at score ≥ 4 to raise the mean toward 4.0.
- Reduce Evol frequency to every 10th round; current 1-in-5 drifts the mix.
```

### 2.7 Design rationale (decision table)

| Choice | Why | Alternative | Trade-off |
|---|---|---|---|
| Seeds authored by hand | Anything else contaminates step one | Reuse an open seed pool | Would import that pool's biases and license obligations |
| Three seeds + three pool | Seeds anchor diversity; pool echoes current mix | Only seeds, or only pool | Seed-only stagnates; pool-only drifts to teacher mode |
| Hash + record the generation prompt | Minimum unit of reproducibility | Free-text in notes | Un-auditable; a prompt tweak later invalidates provenance |
| Cap `MAX_CALLS` | Runaway loops are a realistic failure mode | No cap | Found cheap teachers can still run to large bills |
| Route refusals, not drop | Some are safety-training material for mod-106 | Drop refusals | Loses the only in-distribution refusal signal |
| ROUGE-L for dedup | Catches surface-level near-duplicates cheaply | MinHash Jaccard | MinHash cheaper at scale; ROUGE more interpretable for 200 rows |
| Light Evol cadence | Evol amplifies teacher mistakes | Evol every round | Harder on average, but FP rate spikes without baseline control |

## 3. Validation steps

1. **Call-count reconciliation.** Sum of `raw_generations.jsonl` lines plus Evol rewrites equals `calls` in the pipeline log. If they do not match, teacher responses were cached in a way that bypassed the log (fix the cache layer, not the log).
2. **Drop rate recomputation.** `wc -l raw_generations.jsonl drops/*.jsonl refusals.jsonl synthetic.jsonl`. The numbers in `synth_report.md` must match within one.
3. **Provenance completeness.** `jq 'select(.license == null or .license == "" or .license == "unknown" or .generator == null or .generator_seed_ids == null or (.generator_seed_ids | length) == 0)' synthetic.jsonl` must be empty.
4. **Dedup sanity.** For 20 random pairs of rows in the final set, compute pairwise ROUGE-L; max should be below the threshold (0.7). If any pair crosses, the dedup pool was not updated in order.
5. **License round-trip.** Open `teacher_license.md`, check the model id matches `generator` on every row, and that `license` strings match the terms described in the license doc.
6. **Hand-rate reproducibility.** Have a second reviewer rate the same 10 rows blind; absolute delta on the mean should be ≤ 0.5. Larger means the rubric is under-specified.

## 4. Rubric

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **License decision** | No doc, or license string is "unknown" | Doc exists but one of the four questions skipped | All four questions answered; alternatives considered; `pending-citation` placeholders only where honestly unresolved |
| **Pipeline** | Loop runs but has no filters, or cannot reach 200 | Reaches 200 but one of {dedup, refusal, format} is missing | Reaches 200, three filters wired, raw log + drop log + refusal file all emitted |
| **Provenance** | Rows missing `generator`, `license`, or `generator_seed_ids` | Fields present but hashed prompt omitted | Full envelope on every row; prompt hash matches the generation prompt bytes |
| **Report** | No report, or numbers do not reconcile | Report exists but hand-rating skipped | ≤2 pages, per-filter drop rates reconcile, hand-rated 10 survivors with mean + ≤2 count |
| **Evol / iteration critique** | No Evol; no "next iteration" paragraph | Evol present but not reflected in distribution analysis | Evol tagged in provenance, category drift linked to Evol cadence, concrete next-iteration change |

**Pass threshold:** 7/10, no single dimension at 0.

## 5. Common mistakes

- Generating 200 rows with no deliberate dedup → the pool collapses to a handful of templates the teacher favors.
- Dropping refusals silently. The refusals are the single most interesting training signal the teacher produced.
- Storing `license: "unknown"`. Any row the engineer cannot defend by its license cannot ship. Decide up-front.
- Forgetting to hash the generation prompt. The pipeline is then unreproducible; tomorrow's cost of a prompt tweak is the whole run.
- Running Evol from round 1. Evol amplifies the teacher's errors; it needs a stable baseline underneath it.
- Rating all 10 hand-rated rows at 4. Either the filters are perfect (they aren't) or the reviewer is lenient.
- Using the teacher as its own quality judge. Judges and generators must be different models — otherwise the pipeline is a self-grading student.
- Treating `category` as a free-form string. The mix card (exercise-05) relies on this being a controlled vocabulary.

## 6. References

- [SI23] Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions," ACL 2023. <https://arxiv.org/abs/2212.10560>
- [EI23] Xu et al., "WizardLM: Empowering Large Language Models to Follow Complex Instructions," 2023. (Evol-Instruct) <https://arxiv.org/abs/2304.12244>
- [ALPACA] Taori et al., "Stanford Alpaca," 2023. <https://crfm.stanford.edu/2023/03/13/alpaca.html>
- [L3-LIC] Meta, Llama 3 Community License (via the Llama 3 model card). <https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct>
- [NEMOTRON] NVIDIA, Nemotron-4 model card + NVIDIA Open Model License. <https://huggingface.co/nvidia/Nemotron-4-340B-Instruct>
- OWASP Machine Learning Security Top 10 — ML-02 Data Poisoning, ML-06 AI Supply Chain. <https://owasp.org/www-project-machine-learning-security-top-10/>
- MITRE ATLAS — Training Data Poisoning and Supply Chain Compromise. <https://atlas.mitre.org/>
- NIST AI RMF — Govern/Map; dataset documentation and provenance obligations. <https://www.nist.gov/itl/ai-risk-management-framework>
- SPDX License List. <https://spdx.org/licenses/>
