# Evaluation Math

A theory-and-math reference for **how the RAG pipeline is measured**. Five families of
metrics, plus the ablation experiment. This is a conceptual/formula reference, not a code
walkthrough.

> Companion document: **[ARCHITECTURE.md](./ARCHITECTURE.md)** — the pipeline being measured
> (BM25, dense retrieval, the RRF re-ranking, RBAC, the SQL path).

Formulas are in LaTeX (renders on GitHub) with a plain-English restatement each. All
constants are the actual values used in the code. Every reported metric is rounded to 4
decimal places.

---

## 0. Setup: what is compared

Evaluation pairs a **gold dataset** against the system's **predictions**, aligned by a
shared question index $i$:

| Gold item (`eval_dataset`) | Prediction (`eval_results`) |
|---|---|
| `question` | `response` (the LLM's answer) |
| `ground_truth` (reference answer) | `chunks_retrieved` (child ids returned) |
| `keywords` (required terms, descriptive Qs) | `tables_queried` (SQL tables touched) |
| `answer_type` (`literal` / `descriptive`) | |
| `expected_contexts` (gold chunk ids / table names) | |

Each metric is computed per question, then **averaged** over the questions it applies to.
The metrics fall into three groups by *what* they judge:

- **Retrieval** (§1) — did we fetch the *right context*? (ids only, no LLM text)
- **Answer quality vs. reference** (§2–3) — is the *answer* right? (semantic + keyword)
- **Answer quality vs. itself / context** (§4–5) — is the answer *grounded* and *on-topic*?
  (LLM-as-judge)

---

## 1. Retrieval metrics

**Concept.** Treat retrieval as set/ranking matching: compare the **ids the system retrieved**
against the **gold relevant ids**, ignoring the generated text entirely. Let

- $R_k$ = the top-$k$ retrieved ids (chunks + tables), in rank order,
- $G$ = the set of gold relevant ids.

Only questions with at least one gold id ($|G|\ge 1$) are scored. `k = None` means "use the
whole retrieved list."

### Precision@k — *of what we fetched, how much was relevant*
$$\text{Precision@}k=\frac{|R_k\cap G|}{|R_k|}$$
Denominator is the actual number retrieved (which may be < k); returns 0 if nothing was
retrieved.

### Recall@k — *of what was relevant, how much we fetched*
$$\text{Recall@}k=\frac{|R_k\cap G|}{|G|}$$
Denominator is always the gold set size; returns 0 if there are no relevant ids.

### Hit-rate@k — *did we get at least one right*
$$\text{HitRate@}k=\mathbb{1}\big[\,|R_k\cap G|\ge 1\,\big]$$
1 if any relevant id appears in the top-$k$, else 0. (A lenient "did retrieval work at all.")

### MRR — *how high the first correct hit sits*
Per query, the **reciprocal rank** is the inverse of the 1-indexed position of the *first*
relevant id (0 if none appears):
$$\text{RR}=\frac{1}{\text{rank of first relevant}},\qquad \text{MRR}=\frac{1}{|Q|}\sum_{q\in Q}\text{RR}_q$$
Rewards putting a correct chunk near the top. First hit at rank 1 → 1.0; rank 2 → 0.5; etc.

### NDCG@k — *rank quality vs. the ideal ordering*
Relevance is **binary** ($d_i\in G$ → 1, else 0), with a $\log_2$ rank discount.
$$\text{DCG@}k=\sum_{i=1}^{k}\frac{\mathbb{1}[d_i\in G]}{\log_2(i+1)}
\qquad
\text{IDCG@}k=\sum_{i=1}^{\min(|G|,\,k)}\frac{1}{\log_2(i+1)}$$
$$\text{NDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k}\quad(\,=0\text{ if IDCG}=0)$$
*In words:* DCG sums a discounted reward for every relevant id, where deeper ranks are worth
less ($1/\log_2(i+1)$). IDCG is the DCG of the *perfect* ordering (all relevant ids first).
Their ratio normalizes to $[0,1]$, so NDCG = 1 means "relevant items are ranked as high as
they possibly could be."

**Aggregation.** Each of the five is averaged over the scored questions; the suite is
typically reported at several cutoffs ($k\in\{1,3,5,\text{all}\}$).

---

## 2. Semantic similarity (answer vs. reference)

**Concept.** A correct answer can be worded differently from the reference, so compare them
by **meaning**: embed both `ground_truth` and `response`, take cosine similarity.

$$\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}\in[-1,1]$$

Reported as a normalized $[0,1]$ score so it reads like a grade:

$$s=\frac{\cos(a,b)+1}{2}\in[0,1]$$

*In words:* cosine measures the angle between the answer and reference embeddings; the
rescale maps $-1\!\to\!0$, $0\!\to\!0.5$, $1\!\to\!1$. The evaluation reports the mean cosine
and the mean normalized score over all questions that have both a reference and a response.
(A zero vector yields 0 by guard.)

---

## 3. Keyword coverage (answer vs. required terms)

**Concept.** For descriptive answers, check that the response actually contains the
**required terms** (`keywords`). Coverage is the fraction present:

$$\text{coverage}=\frac{\#\{\text{keywords matched}\}}{\#\{\text{keywords}\}}\in[0,1]$$

(undefined / skipped when a question has no keywords). Averaged over keyword-bearing
questions. "Matched" has two modes:

### Exact
Case-insensitive **substring** test: the keyword string appears verbatim in the response.
Strict and free (no embeddings), but brittle to synonyms, word order, and morphology.

### Fuzzy (cosine over sliding word-windows)
A keyword counts as present if it is semantically close to *some local span* of the answer:

$$\text{match}_\text{fuzzy}(\text{kw})=\Big[\ \max_{w\,\in\,\text{windows}(\text{answer})}\cos(\text{kw},\,w)\ \ge\ \tau\ \Big]$$

with threshold **$\tau = 0.6$**. The windows are **sliding word-spans sized near the keyword's
own length**: if the keyword has $n$ words, the answer is cut into every span of length $n$ to
$n+\text{slack}$ words (**slack = 2**). A verbatim occurrence short-circuits to similarity
**1.0** (so fuzzy is a strict superset of exact).

*Why windows?* A short keyword compared against the *whole* answer gets **diluted** — the
many unrelated words drag the cosine down. Comparing it against spans of roughly its own size
keeps the signal sharp, and the $\max$ asks "does the keyword strongly match *anywhere*?"

> **Known limitation.** Because windows are sized to the keyword and only widen *upward*
> (by `slack`), a keyword whose concept words are **scattered with filler** can be missed —
> e.g. keyword `"ML engineers"` vs answer `"engineers who excel at ML"`: no 2–4-word window
> holds both *engineers* and *ML*, so the best window scores below 0.6. Adjacent re-wordings
> (`"machine learning engineers"`) are caught; dispersed ones may not be.

### Numeric-exact guard
Any keyword **containing a digit** (e.g. `94%`, `22°C`, `ISO 27001`) is matched **exactly**
even in fuzzy mode. Cosine cannot separate close values — `95%` and `96%` embed almost
identically — so allowing fuzzy there would count a wrong-but-close number as correct. The
guard forces those keywords back to substring matching.

| Constant | Value |
|----------|-------|
| Fuzzy threshold $\tau$ | 0.6 |
| Window slack | 2 words |
| Verbatim short-circuit | similarity = 1.0 |
| Digit-bearing keyword | forced exact |

---

## 4. Faithfulness (LLM-as-judge) — *is the answer grounded?*

**Concept.** A faithful answer asserts nothing that the **retrieved context** doesn't
support — this is the hallucination check. Computed in two LLM steps:

1. **Decompose** the answer into a list of **atomic factual claims**.
2. **Judge** each claim: can it be inferred *solely* from the context? (yes/no, context only,
   no outside knowledge.)

$$\text{faithfulness}=\frac{\#\{\text{claims supported by context}\}}{\#\{\text{total claims}\}}\in[0,1]$$

*In words:* the share of the answer's claims that the context actually backs. 1.0 = every
claim is grounded; 0.5 = half the answer is unsupported (partial hallucination). Edge cases:
if there is no answer/context, or no claims could be extracted, the question is skipped
(undefined). The judge is **`gpt-4.1-mini` at temperature 0** (deterministic grading).

---

## 5. Answer relevancy (LLM-as-judge) — *does it address the question?*

**Concept.** An answer can be perfectly grounded yet off-topic. To test relevance, run the
answer "backwards": ask an LLM to generate the **questions this answer would answer**, then
see how close those are to the *original* question.

1. Generate **$n = 3$** distinct questions from the answer (and a `noncommittal` flag).
2. Embed them and the original question; average the cosine similarities:

$$\text{relevancy}=\frac{1}{n}\sum_{i=1}^{n}\cos\big(q_{\text{orig}},\,q_i^{\text{gen}}\big)$$

*In words:* if the answer is on-topic, the questions it implies will resemble the question we
actually asked, so the mean cosine is high. If the answer is **noncommittal** (evasive /
"I don't know") or no questions are generated, relevancy is **0**. Generator is
**`gpt-4.1-mini` at temperature 0**.

---

## 6. Retrieval ablation & weight sweep

These are **experiments built on §1's metrics**, used to check whether the hybrid design (and
its weighting) actually earns its keep. Both run only over **chunk questions** (table-only
questions go through SQL, not the retrievers, so they'd score 0 for everyone and add noise).

### Ablation — does each retrieval path pull its weight?
Run the **same** questions through three retrievers in isolation and score each with the §1
metrics at cutoffs $k\in\{1,3,5,\text{all}\}$:

$$\{\text{bm25-only},\ \text{dense-only},\ \text{hybrid}\}\ \xrightarrow{\ \text{§1 metrics}\ }\ \text{per-metric winner}$$

The winner per metric is just $\arg\max$ over the three paths. This isolates the retrieval
layer (no LLM, no answer generation) so you can see, e.g., whether the dense path beats BM25
and whether fusing them beats either alone.

### Weight sweep — what fusion weight is best?
Rebuild the ensemble across the BM25 weight grid $w_{\text{bm25}}\in\{0.0,0.1,\dots,1.0\}$
(with $w_{\text{sem}}=1-w_{\text{bm25}}$), score at $k=3$, and report the best weight per
metric:

$$\text{best}(m)=\arg\max_{w_{\text{bm25}}}\ \text{metric}_m\big(\text{hybrid}_{[w_{\text{bm25}},\,1-w_{\text{bm25}}]}\big)$$

This probes whether the default **0.4 / 0.6** split (see
[ARCHITECTURE.md §7](./ARCHITECTURE.md#7-hybrid-fusion--the-rrf-re-ranking)) is actually
optimal for this corpus.

> **Subtlety:** $w_{\text{bm25}}=0$ is **not** the same as "dense-only" from the ablation.
> The ensemble still **unions both candidate sets** — it only changes how they're
> RRF-weighted — so $w_{\text{bm25}}=0$ means "rank the union by the dense list," not
> "ignore BM25's candidates."

---

## 7. Practical note — caching (why re-runs are free)

Two caches keep evaluation from re-spending API calls, which is why these metrics can be
recomputed cheaply:

- **Embedding cache** — content-addressed by `md5(text) → vector`. Every text (references,
  responses, keyword windows, generated questions) is embedded at most once and reused across
  the semantic, keyword-fuzzy, and answer-relevancy metrics.
- **LLM-judge cache** — the faithfulness/answer-relevancy results are written to a JSON file;
  if it exists, the metrics are loaded from it with **zero** LLM calls. Deleting the JSON
  forces a recompute.

---

## Metric cheat-sheet

| Metric | Question it answers | Formula (essence) | Range |
|--------|--------------------|--------------------|-------|
| Precision@k | Fetched stuff relevant? | $\lvert R_k\cap G\rvert/\lvert R_k\rvert$ | 0–1 |
| Recall@k | Got all relevant? | $\lvert R_k\cap G\rvert/\lvert G\rvert$ | 0–1 |
| Hit-rate@k | Got any relevant? | $\mathbb{1}[\,R_k\cap G\neq\varnothing\,]$ | 0/1 |
| MRR | First hit how high? | mean $1/\text{rank}_{\text{first}}$ | 0–1 |
| NDCG@k | Ranked near-ideal? | $\text{DCG}/\text{IDCG}$ | 0–1 |
| Semantic | Answer means the same? | $(\cos+1)/2$ | 0–1 |
| Keyword coverage | Required terms present? | matched / total | 0–1 |
| Faithfulness | Answer grounded? | supported claims / total | 0–1 |
| Answer relevancy | Answer on-topic? | mean $\cos(q,\,q^{\text{gen}})$ | 0–1 |
