# mod-111 · exercise-04 — Fine-Tune vs. RAG vs. Prompt Decision Doc — Solution

## 1. Solution overview

The deliverable is a **decision document** — a reusable template and three worked instantiations — that answers "should we fine-tune, retrieve, or prompt-engineer?" for a concrete capability. The decision document is the artifact the team commits; the three worked examples prove the template is specific enough to produce different, defensible answers for different problems.

Four ideas carry the exercise:

- **The three approaches are not substitutes.** Fine-tuning teaches *behavior and format*; RAG injects *fresh or private knowledge*; prompt engineering shapes *one turn's response*. They stack. The most common correct answer is "prompt + RAG now; fine-tune later when the behavior gaps are specific and recurring."
- **The decision is driven by the failure mode, not the technology.** Fine-tune when the base model *knows* the answer but *refuses the format or style*; RAG when the base model *does not know* the fact; prompt-engineer when the base model can do the task given the right scaffolding. The deliverable starts with a failure-mode audit, not a technology comparison.
- **Cost is multi-dimensional.** Build cost (annotation, training, indexing), latency (RAG adds hops; fine-tune adds nothing at inference), token cost (RAG inflates every call; a fine-tune can shrink the prompt), governance cost (fine-tune produces a new weight artifact; RAG lives and dies with the index).
- **Reversibility matters.** A fine-tune bakes behavior into weights and requires a new release cycle to fix; a retrieval index can be rebuilt in a day; a prompt change is a config push. Start with the most reversible intervention that works.

The worked example applies the template to three capabilities for the customer-support-assistant: (A) consistent response format and tone, (B) up-to-date product documentation lookup, (C) refusing out-of-policy refund requests.

## 2. Implementation — worked decision document

### 2.1 The decision template — `decision_template.md`

```markdown
# Capability decision — <capability name>

## 1. Capability statement
One sentence. The thing the product must do that it cannot do today.

## 2. Failure-mode audit
Collect 50-200 production (or realistic synthetic) prompts and label each
with its failure mode:
- F1 KNOWLEDGE_GAP — base model does not know the fact.
- F2 STALE_KNOWLEDGE — base model knows an older version of the fact.
- F3 PRIVATE_KNOWLEDGE — the fact is not public.
- F4 FORMAT_DRIFT — base model answers correctly but in the wrong shape.
- F5 STYLE_DRIFT — base model answers correctly but in the wrong voice.
- F6 POLICY_DRIFT — base model does the wrong thing on policy edges (refusal,
  escalation, PII handling).
- F7 CAPABILITY_GAP — base model cannot do the underlying task (math,
  tool-use, structured extraction) even with scaffolding.
- F8 CONTEXT_OVERFLOW — the base model can do it but the prompt would be too
  long to serve cheaply.
- F9 AMBIGUOUS — multiple of the above.

Freeze the counts before considering interventions.

## 3. Candidate interventions
For each of fine-tune / RAG / prompt engineering (and combinations),
describe:
- Which failure modes it addresses.
- What it does not address.
- Build cost (annotation, training, indexing, prompt-dev).
- Inference-time cost (latency, tokens, hops).
- Reversibility (how long to roll back).

## 4. Trade-off matrix
A table scoring each intervention against the use case's constraints:
coverage of top failure modes, cost, latency, governance.

## 5. Decision
Name the intervention(s), in order, with the triggers for escalation
(e.g. "prompt + RAG now; fine-tune only when FORMAT_DRIFT > 20% of
production traffic for two consecutive weeks").

## 6. Review schedule
When the decision is revisited and what signal reopens it.
```

### 2.2 Failure-mode cheat sheet

The exercise's load-bearing table — the thing a learner should carry in their head:

| Failure mode | Fine-tune? | RAG? | Prompt? | Reasoning |
|---|---|---|---|---|
| F1 KNOWLEDGE_GAP (public, static) | sometimes — only if the knowledge is small, bounded, and important for every call | **yes** | no | RAG is the cheapest intervention for finite public knowledge; fine-tuning rewrites weights to memorize what an index can serve. |
| F2 STALE_KNOWLEDGE | no — fine-tuning snapshots facts and starts aging immediately | **yes** | no | If the fact changes, fine-tune has to re-ship; RAG re-indexes. |
| F3 PRIVATE_KNOWLEDGE (per-tenant / confidential) | risky — bakes data into shared weights | **yes** | no (except small snippets) | RAG with per-tenant indices is the privacy-preserving default; fine-tuning on private data creates a data-governance problem (deletion, access controls). |
| F4 FORMAT_DRIFT | **yes** | partial — few-shot in context helps but is expensive per-call | partial — few-shot in prompt works until prompt is too long | Fine-tune trades build cost for inference cost and format reliability. |
| F5 STYLE_DRIFT | **yes** | no | partial | Same as F4 — fine-tuning is the correct tool when every call needs the same style. |
| F6 POLICY_DRIFT (refusal, escalation) | **yes** (safety-alignment tuning) | partial (policy injection as retrieved context) | partial (policy in prompt) | Prompt-only policy is easy to jailbreak; fine-tuning is the durable path. |
| F7 CAPABILITY_GAP | sometimes — if the capability is teachable from data | no | sometimes — scaffolding (chain-of-thought, tool-use) often closes the gap first | Try scaffolding before committing to fine-tune; some gaps are base-model capacity, which fine-tuning will not fix. |
| F8 CONTEXT_OVERFLOW | **yes** — a fine-tune lets you shrink the prompt | partial — RAG can replace a long fixed prompt with a short retrieved one | partial — prompt compression / chunking buys time | Use fine-tuning to compress a long "system prompt" into the weights when the prompt is stable. |

The pattern: **fine-tune for durable behavior; RAG for fact freshness and privacy; prompt for one-turn scaffolding and quick fixes.**

### 2.3 Worked example A — "consistent response format and tone"

```markdown
# Capability decision — consistent response format and tone

## 1. Capability statement
Agent must reply in the three-part structure
(Diagnosis → Recommendation → Next step) in English or Spanish,
with no trailing filler and no apology-opening on non-error threads.

## 2. Failure-mode audit (N=200 production turns)
| Failure mode     | Count | Share
|---               |---:   |---:
| F4 FORMAT_DRIFT  | 96    | 48%
| F5 STYLE_DRIFT   | 54    | 27%
| F9 AMBIGUOUS     | 22    | 11%
| F1 KNOWLEDGE_GAP | 14    | 7%
| F6 POLICY_DRIFT  |  8    | 4%
| other            |  6    | 3%

The dominant signal is format + style drift (75% combined). Base model can
usually produce the right content; it struggles to produce it in the right
shape every time.

## 3. Candidate interventions
### I1 — Prompt-engineering only (few-shot system prompt + schema hint)
- Addresses: F4, F5 partially; buys time.
- Build cost: low — one prompt-writer week.
- Inference cost: +300-600 prompt tokens per call (the few-shot examples).
- Reversibility: minutes.

### I2 — RAG only
- Addresses: F1 (if we index tickets with resolution patterns).
- Does not address: F4, F5 — the dominant failure modes here.
- Build cost: medium — index build + retrieval service.
- Inference cost: +1-2 hops, +200-800 retrieved tokens per call.
- Reversibility: hours (index rollback).

### I3 — SFT fine-tune on 6k three-part responses
- Addresses: F4, F5 directly; F6 partially.
- Does not address: F1, F2, F3.
- Build cost: high — 6k curated examples + SFT run + eval cycle (mod-102/103).
- Inference cost: zero (same tokens as base at inference).
- Reversibility: a full release cycle (gate + rollback).

### I4 — Prompt + RAG + fine-tune (stack)
- Each layer addresses its native failure mode; costs stack.

## 4. Trade-off matrix
| Intervention | F4 coverage | F5 coverage | Build cost | Inference cost | Reversibility | Governance
|---           |---          |---          |---         |---             |---            |---
| I1 Prompt    | partial     | partial     | low        | high (tokens)  | fast          | low
| I2 RAG       | none        | none        | medium     | medium (tokens + hop) | medium | medium
| I3 SFT       | high        | high        | high       | low            | slow          | high (weight artifact)
| I4 Stack     | high        | high        | high       | medium         | medium        | high

## 5. Decision
**I1 now; I3 in 6-8 weeks if F4+F5 persist above 40% combined on a
50-sample weekly audit.** Rationale:
- F4+F5 at 75% is the dominant signal. I1 attacks it cheaply; I3 attacks it
  durably.
- A fine-tune today commits to a release cycle and a weight artifact we have
  not yet proven we need. The prompt intervention is reversible in minutes;
  we can measure impact in a week.
- If I1 reduces F4+F5 to < 40% and holds, we skip the fine-tune.
- If I1 holds F4+F5 above 40% for two consecutive weeks, we execute I3.
  The budget for I3 is pre-allocated in exercise-02's reserve.

## 6. Review schedule
- Weekly failure-mode audit (N=50 sampled turns) for the first six weeks.
- Decision is reopened automatically if F4+F5 > 40% for two weeks, or if the
  base model line releases a new revision that the quality-eval shows
  materially changes format behavior.
```

### 2.4 Worked example B — "up-to-date product documentation lookup"

```markdown
# Capability decision — up-to-date product documentation lookup

## 1. Capability statement
Agent must answer "what does feature X do in version Y?" accurately for the
currently-shipping product (docs updated weekly).

## 2. Failure-mode audit (N=150 synthetic + production)
| Failure mode          | Count | Share
|---                    |---:   |---:
| F2 STALE_KNOWLEDGE    | 78    | 52%
| F3 PRIVATE_KNOWLEDGE  | 36    | 24%  (internal-only feature docs)
| F1 KNOWLEDGE_GAP      | 22    | 15%
| F9 AMBIGUOUS          |  9    |  6%
| other                 |  5    |  3%

The dominant failure mode is "the base model is confidently wrong about the
current version because it is working from an older snapshot."

## 3. Candidate interventions
### I1 — Prompt-engineering (paste docs into system prompt)
- Addresses: F1-F3 per call, but only for the subset the prompt can hold.
- Build cost: low.
- Inference cost: very high (docs are 10k+ tokens).
- Reversibility: fast.
- Fatal: scale. The docs will not fit per call.

### I2 — RAG
- Addresses: F1, F2, F3 directly.
- Build cost: medium — index build, embeddings, retrieval service,
  tenant-scoped access controls for private docs.
- Inference cost: +1 retrieval hop, +500-1500 retrieved tokens per call.
- Reversibility: hours (re-index) or seconds (feature-flag off the retrieval).

### I3 — SFT fine-tune on current docs
- Addresses: F1 and F3 only; F2 ages out immediately.
- Build cost: high; the "data" is a moving target.
- Inference cost: zero.
- Reversibility: a release cycle per docs update — infeasible at weekly cadence.

## 4. Trade-off matrix
| Intervention | F2 coverage | F3 coverage | Build cost | Inference cost | Reversibility
|---           |---          |---          |---         |---             |---
| I1 Prompt    | partial     | partial     | low        | fatal          | fast
| I2 RAG       | high        | high        | medium     | medium         | hours
| I3 SFT       | ages out    | partial     | high       | zero           | release cycle

## 5. Decision
**I2 (RAG with per-tenant scoping).** Rationale:
- F2 at 52% means every fine-tune we ship starts out of date within a week
  of docs updates. RAG handles freshness by re-indexing.
- F3 (24%) requires per-tenant retrieval scopes — RAG's index-level access
  controls are the privacy-preserving mechanism. A fine-tune on private
  tenant data creates a shared-weight-with-private-data governance problem
  (deletion requests, per-tenant audits).
- No reason to stack a fine-tune on top yet; the base model's instruction
  following on the retrieved context is adequate.

## 6. Review schedule
- Weekly retrieval-quality audit (recall@5, MRR against a frozen 200-query
  eval set).
- Monthly check on F4/F5 in the audit — if format drift grows while
  retrieval is healthy, fine-tune enters consideration (per worked example A).
- Decision reopens if the docs stop changing for 6+ months (fine-tune might
  then become competitive on inference cost).
```

### 2.5 Worked example C — "refuse out-of-policy refund requests"

```markdown
# Capability decision — refuse out-of-policy refund requests

## 1. Capability statement
Agent must decline refund requests that fall outside the written refund
policy, explain *why* using the published policy, and escalate to a human
when the customer disputes.

## 2. Failure-mode audit (N=100 adversarial + production)
| Failure mode     | Count | Share
|---               |---:   |---:
| F6 POLICY_DRIFT  | 61    | 61%
| F5 STYLE_DRIFT   | 18    | 18%  (refuses in the wrong tone)
| F4 FORMAT_DRIFT  | 12    | 12%  (missing policy citation)
| other            |  9    |  9%

The dominant mode is policy-drift: the base model either grants refunds it
should not (helpful-to-a-fault) or refuses safe adjacent requests
(over-refusal).

## 3. Candidate interventions
### I1 — Prompt with policy text + examples
- Addresses: F6 partially; vulnerable to prompt-injection ("ignore previous
  instructions...") and to tone-overrule from the customer.
- Build cost: low.
- Inference cost: +400-800 prompt tokens per call.
- Reversibility: fast.

### I2 — RAG with retrieved policy clause per request
- Addresses: F6 partially — same vulnerabilities as I1 plus retrieval misses.
- Build cost: medium.
- Reversibility: hours.

### I3 — Safety-alignment tuning (SFT + DPO on refusal + policy-citation data)
- Addresses: F6 and F5 directly; makes refusal behavior resistant to prompt
  injection because it is in the weights, not in the context.
- Build cost: high — mod-106 pipeline (refusal data + over-refusal eval +
  red-team + DPO).
- Inference cost: zero.
- Reversibility: release cycle; must co-ship with the kill switch from
  exercise-03.

### I4 — Stack: I3 + I2 (retrieve the specific policy clause to cite)
- Fine-tune carries the refusal behavior; RAG carries the exact clause text.
- Each layer addresses its native failure mode.

## 4. Trade-off matrix
| Intervention | F6 coverage | Injection-resistant | Build cost | Governance
|---           |---          |---                  |---         |---
| I1 Prompt    | partial     | no                  | low        | low
| I2 RAG       | partial     | no                  | medium     | medium
| I3 SFT+DPO   | high        | yes                 | high       | high
| I4 Stack     | high        | yes                 | high       | high

## 5. Decision
**I4 (stack: safety-alignment tuning + per-clause retrieval).** Rationale:
- F6 at 61% is both a product risk (giving refunds that should be denied) and
  a safety concern (the inverse: refusing safe customers in the wrong tone).
  The durable intervention is in the weights.
- Prompt-only policy is routinely jailbroken; a safety-aligned model + RAG
  for clause citation is the pattern the mod-106 lessons recommend.
- The build cost is high but the data overlaps substantially with exercise-02's
  safety bucket; the marginal ask is the policy-citation RAG index.

## 6. Review schedule
- Daily red-team sample (10 adversarial refund prompts) for the first two
  weeks post-launch; weekly after.
- Reopen if the refund policy changes materially (new clauses, new
  escalation paths) — RAG index refreshes do not re-trigger the decision;
  weight-level refusal-behavior changes do.
```

### 2.6 Combined decision summary

| Capability | Dominant failure mode | Decision | First intervention | Later intervention |
|---|---|---|---|---|
| A. Response format and tone      | F4 + F5 (75%) | prompt now, fine-tune later | prompt few-shot + schema | SFT if F4+F5 > 40% for 2 weeks |
| B. Documentation lookup          | F2 + F3 (76%) | RAG only                     | retrieval with per-tenant scopes | revisit if docs stop changing |
| C. Refund policy refusal         | F6 (61%)      | fine-tune + RAG              | safety-alignment tuning + clause retrieval | iterate on red-team findings |

A and B and C get different answers because their failure modes are different. The template produced defensible decisions, not a universal verdict.

### 2.7 Directory layout (as delivered)

```
outputs/exercise-04/
├── decision_template.md
├── failure_mode_cheatsheet.md
├── decisions/
│   ├── capability_A_format_and_tone.md
│   ├── capability_B_docs_lookup.md
│   └── capability_C_refund_policy.md
├── audits/
│   ├── capability_A_audit_N200.jsonl    # labeled turns feeding §2 of each decision
│   ├── capability_B_audit_N150.jsonl
│   └── capability_C_audit_N100.jsonl
└── README.md
```

## 3. Validation steps

1. **The decision starts with a failure-mode audit, not a technology comparison.** Each decision doc's §2 cites a sample size, a label distribution, and a frozen file under `audits/`. A decision that leads with "we should fine-tune" without a labeled audit is motivated reasoning.
2. **Sample size supports the claim.** N=50 is a scoping audit; N=150-200 is a decision audit. Decisions that lean on N=15 have CIs wide enough to flip the verdict and must be re-sampled.
3. **Every candidate intervention names what it does *not* address.** Fine-tune does not fix stale knowledge; RAG does not fix format drift; prompt does not durably fix policy. A decision doc that lists only upsides is incomplete.
4. **Cost is reported along all four axes.** Build cost, inference cost (tokens and hops), latency, governance. A decision that compares interventions on build cost alone misses the production bill.
5. **Reversibility is named.** Each intervention has a time-to-revert. The default rule (start with the most reversible intervention that works) is testable only if reversibility is quantified.
6. **Decision is action-conditioned.** Not "we might fine-tune later" but "we fine-tune when metric X exceeds threshold Y for duration Z." The trigger is the testable thing.
7. **Reviewers converge.** Two reviewers produce decisions that match in intervention *and* trigger on the same audit. Divergence is template ambiguity, not reviewer preference.
8. **Review schedule is present.** A decision without a review schedule is a decision made once; the template requires a weekly/monthly cadence and an automatic reopen clause.
9. **Private data flows are audited before SFT is picked.** If F3 (private knowledge) is dominant and SFT is the chosen intervention, a governance review is required before committing (data-residency, deletion, cross-tenant exposure).
10. **Combined stack decisions explain the composition.** Decision C stacks SFT and RAG; the doc explains which layer carries which failure mode. A stack without decomposition is not defensible.

## 4. Rubric for grading the deliverable

Grade on five dimensions, 2 points each, 10 total.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| **Failure-mode audit** | Audit absent or unlabeled | Audit present but sample size undersupplies the claim | N ≥ 150 for decision audits; frozen jsonl; label distribution per decision; dominant failure mode drives the recommendation |
| **Intervention coverage** | Only one approach considered | Two approaches, incomplete coverage | All three (prompt / RAG / fine-tune) + at least one stack option; each annotated with what it does NOT address |
| **Cost + reversibility accounting** | One-dimensional cost | Build + inference cost | All four: build, inference (tokens + hops), latency, governance; reversibility quantified (minutes/hours/release-cycle) |
| **Action-conditioned decision** | "We might fine-tune later" | Triggers named but not measurable | Explicit trigger (metric, threshold, duration) for every escalation; reserve budget (from exercise-02) tied to the trigger |
| **Reusability of the template** | Hard-coded to the worked example | Template extracted but uses only one example | `decision_template.md` + ≥3 worked instantiations producing different answers; failure-mode cheat sheet referenced |

**Pass threshold:** 7/10, with no single dimension at 0 and *failure-mode audit* at 2.

## 5. Common mistakes

- **Starting from the technology.** "We want to fine-tune" is not a decision input. The decision input is the failure-mode distribution; the technology is downstream.
- **Treating RAG as free.** Retrieval adds a hop, inflates every call by hundreds of tokens, and introduces index-staleness as a new failure mode. "RAG as default" without an inference-cost line is a tell.
- **Fine-tuning on moving data.** The classic mistake: a fine-tune on documentation that updates weekly starts rotting the week after launch. The decision for F2 (stale knowledge) is RAG; only commit to fine-tune when the data is stable.
- **Prompt engineering as an eternal answer.** Prompts past 2k tokens that have accumulated for months are a sign that fine-tuning would be cheaper at inference and easier to maintain. The trigger ("prompt > 2k tokens *and* F4+F5 dominant") exists to notice this.
- **Baking private data into shared weights.** F3 + SFT without a governance review is a deletion-and-access-control nightmare later. RAG with per-tenant scopes is the privacy-preserving default.
- **Stacking without decomposition.** "Fine-tune + RAG + prompt" with no per-layer failure-mode assignment is a stack of hope. Name the layer that owns each failure mode.
- **Confusing capability gap for behavior gap.** F7 (capability gap) often is not fixed by fine-tuning — if the base model cannot do structured extraction at all, fine-tuning to make it try harder will plateau. Try scaffolding (schema prompting, tool-use) before paying the fine-tune cost.
- **No audit refresh.** The decision is made on audits from month zero; six months later, the failure distribution has shifted and the decision has not. The review schedule exists to catch this.
- **Decision doc as narrative.** A decision that cannot be reduced to an action-triggered rule ("fine-tune when X > Y for Z") is not operational. Narrative is the explanation; the rule is the decision.
- **Ignoring jailbreak exposure on policy.** Prompt- or RAG-only policy is routinely bypassed; a fine-tune carries policy into weights where context attacks cannot overrule it. F6 dominant → fine-tune dominant.
- **No budget linkage.** The escalation trigger ("fine-tune in 6-8 weeks") must map to the dataset-budget reserve from exercise-02. A trigger that fires with no budget is a wish.
- **Picking a technology because the team knows it.** Familiarity is a real cost consideration but is accounted for under *build cost*; it is not a dominant axis.

## 6. References

- NIST AI Risk Management Framework — *Map-2 (categorization and context)*, *Measure-2.11 (fairness and bias)*, *Manage-1.3 (risk prioritization)*, and *Govern-3.2 (accountability for decisions)* frame the decision document as a governed artifact. <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF Generative AI Profile — explicit guidance on when retrieval vs. fine-tuning vs. prompting is the appropriate mitigation for generative-AI risks. <https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/>
- OWASP Machine Learning Security Top 10 — *ML02 Data Poisoning*, *ML03 Model Inversion*, and *ML05 Model Theft* back the governance-cost treatment of fine-tuning on private data. <https://owasp.org/www-project-machine-learning-security-top-10/>
- OWASP LLM Top 10 for Large Language Model Applications — *LLM01 Prompt Injection*, *LLM06 Sensitive Information Disclosure*, and *LLM08 Excessive Agency* motivate the injection-resistance argument for fine-tuning F6 (policy drift). <https://owasp.org/www-project-top-10-for-large-language-model-applications/>
- MITRE ATLAS — *AML.T0051 LLM Prompt Injection*, *AML.T0057 LLM Data Leakage*, and the retrieval-augmented-generation entries ground the threat model behind the cost-of-RAG-and-private-data discussion. <https://atlas.mitre.org/>
- Lewis et al. — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (original RAG formulation). <https://arxiv.org/abs/2005.11401>
- Ouyang et al. — *Training language models to follow instructions with human feedback* (InstructGPT; the behavioral-tuning baseline). <https://arxiv.org/abs/2203.02155>
- Rafailov et al. — *DPO* (preference optimization reference used in decision C). <https://arxiv.org/abs/2305.18290>
- Wei et al. — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (scaffolding-first path for F7 CAPABILITY_GAP). <https://arxiv.org/abs/2201.11903>
- Hugging Face — *RAG, fine-tuning, and prompting: when to use which* guidance. <https://huggingface.co/docs/transformers/main/en/tasks/retrieval_augmented_generation>
- LangChain / LlamaIndex documentation — practitioner references for the RAG reference architectures discussed in decision B. <https://python.langchain.com/docs/tutorials/rag/> · <https://docs.llamaindex.ai/>
