"""Load (or build) a document's eval dataset and eval results as two dicts.

Usage:
    from main import DATA_DIR
    from test_eval import load_eval_data
    eval_dataset, eval_results = load_eval_data(DATA_DIR / "Employee Performance.docx")

Both dicts are keyed by a shared integer index (0, 1, 2, …) and restricted to the
questions present in BOTH files, so eval_dataset[i] and eval_results[i] always refer
to the SAME question (matched on the question string internally, then re-keyed by
index). If eval_results only has 3 answered questions, eval_dataset is trimmed to the
same 3. The question text stays inside each value (item["question"] / rec["question"]):
    eval_dataset[i] -> gold item   {question, ground_truth, answer_type, keywords, expected_contexts}
    eval_results[i] -> eval record {question, response, chunks_retrieved, tables_queried, sources}

No redundant work / API calls: if processed/<hash>/eval_dataset.json and
eval_results.json already exist, they are just loaded. Otherwise only the missing
one is generated — the dataset via main.build_agent(generate_eval=True), the results
via eval.run_eval (which itself generates the dataset too if it is missing).
"""

import json
import math
import os
import hashlib
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel

from main import build_agent, file_hash, PROCESSED_DIR, DATA_DIR
from eval import run_eval


def load_eval_data(file_path, role="manager", results_name="eval_results.json"):
    """Return (eval_dataset, eval_results) for `file_path` as dicts keyed by a shared
    integer index, so eval_dataset[i] and eval_results[i] refer to the same question.

    Each file is checked independently and ONLY the missing one is generated:
      - eval_dataset.json missing -> build_agent(generate_eval=True) makes just the dataset
      - eval_results.json missing -> run_eval makes just the results (reusing the dataset)
    If both exist, nothing is generated (no API calls). If both are missing, run_eval
    creates the dataset and then the results in a single agent build.

    results_name lets you point at an alternate results file (e.g. "eval_results_small.json")
    without regenerating anything — it must already exist.
    """
    processed_dir = os.path.join(PROCESSED_DIR, file_hash(file_path))
    dataset_path = os.path.join(processed_dir, "eval_dataset.json")
    results_path = os.path.join(processed_dir, results_name)

    have_dataset = os.path.exists(dataset_path)
    have_results = os.path.exists(results_path)

    if not have_dataset and not have_results:
        # neither exists -> run_eval generates the dataset, then the results (one build).
        print("eval_dataset.json + eval_results.json missing -> generating both")
        run_eval(file_path, role=role)
    elif not have_dataset:
        # only the dataset is missing -> generate JUST the dataset.
        print("eval_dataset.json missing -> generating dataset only")
        build_agent(file_path, generate_eval=True, role=role)
    elif not have_results:
        # only the results are missing -> generate JUST the results (dataset is reused).
        print("eval_results.json missing -> generating results only")
        run_eval(file_path, role=role)
    else:
        # both already present -> generate nothing, no API calls.
        print("eval_dataset.json + eval_results.json present -> loading (no generation)")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_items = json.load(f)
    with open(results_path, "r", encoding="utf-8") as f:
        result_records = json.load(f)

    full_results = {rec["question"]: rec for rec in result_records}

    # keep only questions answered in BOTH files (eval_results may be a subset of the
    # dataset, e.g. when run_eval answered only k questions), in the dataset's original
    # order. Match on the question string, then re-key BOTH dicts by a shared integer
    # index so eval_dataset[i] and eval_results[i] always refer to the SAME question.
    common_items = [item for item in dataset_items if item["question"] in full_results]

    eval_dataset = {i: item for i, item in enumerate(common_items)}
    eval_results = {i: full_results[item["question"]] for i, item in enumerate(common_items)}

    return eval_dataset, eval_results


# --- helpers: pull the IDs out of each side ---------------------------------

def _expected_ids(gold_item):
    """The relevant context IDs for one question (chunk_ids + table_names)."""
    ids = set()
    for ctx in gold_item.get("expected_contexts", []):
        ids.add(ctx.get("chunk_id") or ctx.get("table_name"))
    ids.discard(None)
    return ids


def _retrieved_ids(result):
    """What the system actually retrieved, in rank order: chunks first, then tables."""
    return list(result.get("chunks_retrieved", [])) + list(result.get("tables_queried", []))


def precision_at_k(retrieved, relevant, k=None):
    top = retrieved[:k] if k else retrieved
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def recall_at_k(retrieved, relevant, k=None):
    if not relevant:
        return 0.0
    top = retrieved[:k] if k else retrieved
    return len(set(top) & relevant) / len(relevant)


def hit_rate_at_k(retrieved, relevant, k=None):
    top = retrieved[:k] if k else retrieved
    return 1.0 if (set(top) & relevant) else 0.0


def reciprocal_rank(retrieved, relevant):
    """1 / rank of the first relevant hit (0 if none). MRR is the mean of this."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, relevant, k=None):
    cutoff = k if k else len(retrieved)
    top = retrieved[:cutoff]
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(top, start=1) if item in relevant)
    ideal_n = min(len(relevant), cutoff)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


# --- aggregate over the whole set -------------------------------------------

def retrieval_evaluation(validation_set, eval_results, k=None):
    """Average the 5 retrieval metrics over every aligned question.

    validation_set[i] -> gold item (expected_contexts)
    eval_results[i]   -> prediction (chunks_retrieved + tables_queried)
    k: top-k cutoff (None = use everything retrieved).
    """
    totals = {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}
    scored = 0

    for i in validation_set.keys() & eval_results.keys():
        relevant = _expected_ids(validation_set[i])
        if not relevant:                       # no gold context -> can't score retrieval
            continue
        retrieved = _retrieved_ids(eval_results[i])

        totals["precision"] += precision_at_k(retrieved, relevant, k)
        totals["recall"]    += recall_at_k(retrieved, relevant, k)
        totals["hit_rate"]  += hit_rate_at_k(retrieved, relevant, k)
        totals["mrr"]       += reciprocal_rank(retrieved, relevant)
        totals["ndcg"]      += ndcg_at_k(retrieved, relevant, k)
        scored += 1

    if scored == 0:
        return {m: 0.0 for m in totals}
    return {m: round(v / scored, 4) for m, v in totals.items()}


_embedder = OpenAIEmbeddings(model="text-embedding-3-small")


def _embeddings_cache_path(file_path):
    """Per-file embedding cache: processed/<hash>/answer_embeddings.json — it lives and
    dies with that document's processed folder, like its chunks/eval files."""
    return os.path.join(PROCESSED_DIR, file_hash(file_path), "answer_embeddings.json")


def _load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _embed_cached(text, cache):
    """Embed `text`, reusing/adding to the given cache dict (md5(text) -> vector).
    Does NOT write to disk — the caller persists the cache once when it's done."""
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    if key not in cache:
        cache[key] = _embedder.embed_query(text)        # the only API call
    return cache[key]


def _embed_many_cached(texts, cache):
    """Embed many texts in ONE API call for the uncached (deduped) ones, instead of one
    call each — the big token saver for fuzzy keyword matching, which compares a keyword
    against many response windows. Fills `cache` (md5(text) -> vector) and returns the
    vectors aligned to `texts`."""
    keys = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
    missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in cache))
    if missing:
        for t, vec in zip(missing, _embedder.embed_documents(missing)):
            cache[hashlib.md5(t.encode("utf-8")).hexdigest()] = vec
    return [cache[k] for k in keys]


def cosine_similarity(vec_a, vec_b):
    """Cosine of the angle between two vectors. Range: -1 (opposite) .. 1 (identical)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_similarity(text_a, text_b, cache=None):
    """Embed two texts and measure how close they mean.

    Pass a `cache` dict (md5(text) -> vector) to reuse embeddings; omit it for a
    one-off comparison (embeds fresh, nothing persisted).

    Returns (cosine, score):
      cosine -> raw cosine similarity, range -1 .. 1
      score  -> same thing normalized to 0 .. 1  ((cosine + 1) / 2)
    """
    if cache is None:
        cache = {}
    vec_a = _embed_cached(text_a, cache)
    vec_b = _embed_cached(text_b, cache)
    cos = cosine_similarity(vec_a, vec_b)
    return cos, (cos + 1) / 2


def semantic_evaluation(validation_set, eval_results, file_path):
    """Average cosine similarity + semantic similarity score of ground_truth vs response,
    over every aligned question.

    validation_set[i] -> gold item (uses "ground_truth")
    eval_results[i]   -> prediction (uses "response")
    file_path         -> which document this eval is for; its embedding cache lives at
                         processed/<hash>/answer_embeddings.json (per-file, reused on re-runs)
    """
    cache_path = _embeddings_cache_path(file_path)
    cache = _load_cache(cache_path)
    n_before = len(cache)

    total_cos = 0.0
    total_score = 0.0
    scored = 0

    for i in validation_set.keys() & eval_results.keys():
        truth = validation_set[i].get("ground_truth", "")
        answer = eval_results[i].get("response", "")
        if not truth or not answer:           # nothing to compare -> skip
            continue
        cos, score = semantic_similarity(truth, answer, cache)
        total_cos += cos
        total_score += score
        scored += 1

    # persist the per-file cache once, only if we embedded anything new this run
    if len(cache) > n_before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    if scored == 0:
        return {"cosine_similarity": 0.0, "semantic_similarity_score": 0.0}
    return {
        "cosine_similarity": round(total_cos / scored, 4),
        "semantic_similarity_score": round(total_score / scored, 4),
    }


# --- keyword coverage: do the answer's REQUIRED keywords actually show up? --------
# Descriptive items carry a `keywords` list (the essential terms a correct answer must
# contain). keyword_evaluation scores how many of them appear in the generated response.
# Two matching modes:
#   exact -> case-insensitive substring (strict; brittle to synonyms/word-order/morphology)
#   fuzzy -> embedding cosine similarity (lenient: "ML engineers" ~ "machine learning
#            engineers"). Fuzzy is its own function and is a strict SUPERSET of exact
#            (a verbatim keyword short-circuits to similarity 1.0, no embedding needed).


def keyword_match_exact(keyword, text):
    """Exact, case-insensitive substring presence of `keyword` in `text`."""
    return keyword.strip().lower() in text.lower()


def _is_numeric_keyword(keyword):
    """True if the keyword carries a specific VALUE (contains any digit) — e.g. "94%",
    "22°C", "ISO 27001", "Q1 2026". Cosine is unsafe for these (close values embed almost
    identically: "95%" ~ "96%"), so they must match EXACTLY even in fuzzy mode, otherwise
    a wrong-but-close number would count as a false hit."""
    return any(ch.isdigit() for ch in keyword)


def _word_windows(text, n_words, slack=2):
    """Sliding word-windows of `text` sized near a keyword's length (n_words .. n_words +
    slack). A keyword is compared against these LOCAL spans so a short keyword isn't
    diluted by a long answer. Falls back to the whole text when it is shorter than the
    window. .split() (not \\w+) so tokens like "R-14", "94%", "v3.2" stay intact."""
    words = text.split()
    spans = set()
    for size in range(max(1, n_words), n_words + slack + 1):
        if len(words) <= size:
            spans.add(text.strip())
        else:
            for i in range(len(words) - size + 1):
                spans.add(" ".join(words[i:i + size]))
    return [s for s in spans if s]


def keyword_similarity(keyword, text, cache=None):
    """Fuzzy presence score of `keyword` in `text`: the MAX cosine similarity between the
    keyword and any local word-window of the text (~0..1). ~1.0 if the keyword appears
    verbatim, high for synonyms / re-wordings, low if the meaning is absent.

    Pass a `cache` dict (md5(text) -> vector) to reuse embeddings; omit for a one-off.
    NOTE: cosine is unreliable for numbers/codes ("94%" ~ "98%" embed almost identically).
    This is a raw fuzzy primitive — keyword_coverage(numeric_exact=True) is what actually
    routes digit-bearing keywords to exact matching, so don't rely on this alone for them.
    """
    if cache is None:
        cache = {}
    keyword = keyword.strip()
    if keyword.lower() in text.lower():        # verbatim -> present, skip embedding
        return 1.0
    spans = _word_windows(text, len(keyword.split()))
    vecs = _embed_many_cached([keyword] + spans, cache)    # one batched API call
    kw_vec, span_vecs = vecs[0], vecs[1:]
    return max((cosine_similarity(kw_vec, sv) for sv in span_vecs), default=0.0)


def keyword_match_fuzzy(keyword, text, threshold=0.6, cache=None):
    """True if `keyword` is fuzzily present in `text` (keyword_similarity >= threshold)."""
    return keyword_similarity(keyword, text, cache) >= threshold


def _keyword_matched(keyword, response, fuzzy, threshold, cache, numeric_exact):
    """Whether one keyword counts as present, honoring the numeric-exact guard: digit-
    bearing keywords always use exact matching (cosine is unsafe for close values).
    Single source of truth shared by keyword_coverage and keyword_misses."""
    use_fuzzy = fuzzy and not (numeric_exact and _is_numeric_keyword(keyword))
    if use_fuzzy:
        return keyword_match_fuzzy(keyword, response, threshold, cache)
    return keyword_match_exact(keyword, response)


def keyword_coverage(keywords, response, fuzzy=False, threshold=0.6, cache=None,
                     numeric_exact=True):
    """Fraction (0..1) of `keywords` present in `response`; None if there are no keywords.
    Exact substring by default. With fuzzy=True, prose keywords use cosine matching, but
    value-bearing keywords (digits, see _is_numeric_keyword) still match EXACTLY when
    numeric_exact=True — so "95%" can't be satisfied by "96%". Set numeric_exact=False for
    pure fuzzy matching on every keyword."""
    if not keywords:
        return None
    hits = sum(1 for k in keywords
               if _keyword_matched(k, response, fuzzy, threshold, cache, numeric_exact))
    return hits / len(keywords)


def keyword_misses(keywords, response, fuzzy=False, threshold=0.6, cache=None,
                   numeric_exact=True):
    """The required keywords NOT present in `response`, using the SAME match rules as
    keyword_coverage. An empty list means full coverage (no failures)."""
    return [k for k in keywords
            if not _keyword_matched(k, response, fuzzy, threshold, cache, numeric_exact)]


def keyword_evaluation(validation_set, eval_results, fuzzy=False, threshold=0.6,
                       file_path=None, numeric_exact=True):
    """Average keyword coverage over every descriptive (keyword-bearing) question.

    validation_set[i] -> gold item (uses "keywords")
    eval_results[i]   -> prediction (uses "response")
    fuzzy=False -> exact case-insensitive substring match (no API calls).
    fuzzy=True  -> embedding cosine match (>= threshold); needs file_path for the per-file
                   embedding cache (processed/<hash>/answer_embeddings.json, shared with
                   semantic_evaluation so embeddings are reused, not recomputed).
    numeric_exact (fuzzy only) -> value-bearing keywords (digits) still match EXACTLY, so a
                   close-but-wrong number ("96%" for "95%") can't score a false hit.
    """
    cache = {}
    cache_path = None
    if fuzzy:
        if file_path is None:
            raise ValueError("keyword_evaluation(fuzzy=True) needs file_path for the cache")
        cache_path = _embeddings_cache_path(file_path)
        cache = _load_cache(cache_path)
    n_before = len(cache)

    total = 0.0
    scored = 0
    for i in validation_set.keys() & eval_results.keys():
        keywords = validation_set[i].get("keywords") or []
        if not keywords:                       # literal items have no keywords -> skip
            continue
        response = eval_results[i].get("response", "")
        total += keyword_coverage(keywords, response, fuzzy=fuzzy,
                                  threshold=threshold, cache=cache,
                                  numeric_exact=numeric_exact)
        scored += 1

    # persist any newly embedded keywords/windows (fuzzy mode only)
    if fuzzy and cache_path and len(cache) > n_before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    mode = "fuzzy" if fuzzy else "exact"
    if scored == 0:
        return {"keyword_coverage": 0.0, "n_questions": 0, "mode": mode}
    return {
        "keyword_coverage": round(total / scored, 4),
        "n_questions": scored,
        "mode": mode,
    }


def print_keyword_failures(validation_set, eval_results, threshold=0.6, file_path=None,
                           numeric_exact=True):
    """For EXACT then FUZZY mode, print (and return as text) the keywords each answer
    failed to cover — with the ground-truth answer and the LLM's answer for context.
    Prints a 'no failures' line for a mode that missed nothing. Only descriptive
    (keyword-bearing) questions are considered. Fuzzy needs file_path for the embedding
    cache (it embeds any keywords/windows not already cached, in batches)."""
    cache = {}
    cache_path = None
    if file_path is not None:
        cache_path = _embeddings_cache_path(file_path)
        cache = _load_cache(cache_path)
    n_before = len(cache)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    indices = sorted(validation_set.keys() & eval_results.keys())

    def collect(fuzzy):
        failures = {}
        for i in indices:
            keywords = validation_set[i].get("keywords") or []
            if not keywords:                       # literal items have no keywords
                continue
            response = eval_results[i].get("response", "")
            miss = keyword_misses(keywords, response, fuzzy=fuzzy, threshold=threshold,
                                  cache=cache, numeric_exact=numeric_exact)
            if miss:
                failures[i] = miss
        return failures

    for label, fuzzy in [("EXACT", False), ("FUZZY", True)]:
        fails = collect(fuzzy)
        out(f"\n========== {label} mode failures ==========")
        if not fails:
            out("  none - every required keyword was covered.")
            continue
        for i, miss in fails.items():
            item = validation_set[i]
            out(f"\n  Q{i}: {item['question']}")
            out(f"    missed keywords : {miss}")
            out(f"    ground truth    : {item.get('ground_truth', '')}")
            out(f"    llm answer      : {eval_results[i].get('response', '')}")

    # persist any embeddings the fuzzy pass had to compute
    if cache_path and len(cache) > n_before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    return "\n".join(lines)


# --- LLM-judge metrics: faithfulness + answer relevancy (RAGAS-style) ------------
# These call an LLM (gpt-4.1-mini) per question, so they cost tokens — unlike the
# retrieval/keyword/semantic metrics above. faithfulness needs the retrieved context
# (reconstructed from chunks_retrieved -> the PARENT text the agent actually saw).

_judge = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


class _Claims(BaseModel):
    claims: list[str]


class _Verdicts(BaseModel):
    verdicts: list[bool]            # one per claim, in the same order


class _GenQuestions(BaseModel):
    questions: list[str]
    noncommittal: bool             # True if the answer is evasive / "I don't know"


_claim_extractor = _judge.with_structured_output(_Claims)
_claim_judge = _judge.with_structured_output(_Verdicts)
_question_generator = _judge.with_structured_output(_GenQuestions)


def _load_context_maps(file_path):
    """Maps to rebuild the context the agent saw: child chunk_id -> parent_id, parent_id
    -> parent text, and child chunk_id -> child text (fallback)."""
    base = os.path.join(PROCESSED_DIR, file_hash(file_path))
    with open(os.path.join(base, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    parents = []
    parents_path = os.path.join(base, "parents.json")
    if os.path.exists(parents_path):
        with open(parents_path, encoding="utf-8") as f:
            parents = json.load(f)
    child_to_parent = {c["chunk_id"]: c.get("parent_id") for c in chunks}
    child_text = {c["chunk_id"]: c.get("text", "") for c in chunks}
    parent_text = {p["parent_id"]: p.get("text", "") for p in parents}
    return child_to_parent, child_text, parent_text


def _result_context(result, child_to_parent, parent_text, child_text):
    """The context text the agent retrieved for one result: the unique PARENT chunks behind
    its chunks_retrieved (children are swapped to parents, deduped), joined together."""
    parts, seen = [], set()
    for cid in result.get("chunks_retrieved", []):
        pid = child_to_parent.get(cid)
        key = pid or cid
        if key in seen:
            continue
        seen.add(key)
        parts.append(parent_text.get(pid) or child_text.get(cid) or "")
    return "\n\n".join(p for p in parts if p)


def faithfulness(question, answer, context):
    """RAGAS faithfulness: fraction of the answer's atomic claims that the CONTEXT supports.
    Two LLM calls — extract claims, then judge them all at once. Returns None when there
    are no claims or no context (can't be scored). Range 0..1; 1 = no hallucination."""
    if not answer or not context:
        return None
    claims = _claim_extractor.invoke(
        "Break the ANSWER into a list of atomic factual claims it asserts (each a short, "
        "standalone statement). Include only claims the answer actually states.\n\n"
        f"QUESTION: {question}\nANSWER: {answer}"
    ).claims
    if not claims:
        return None
    numbered = "\n".join(f"{n}. {c}" for n, c in enumerate(claims, 1))
    verdicts = _claim_judge.invoke(
        "For EACH claim decide if it can be inferred SOLELY from the context (true) or not "
        "(false). Use only the context, not outside knowledge. Return one verdict per claim "
        f"in the same order.\n\nCONTEXT:\n{context}\n\nCLAIMS:\n{numbered}"
    ).verdicts[:len(claims)]
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v) / len(claims)


def answer_relevancy(question, answer, n=3, cache=None):
    """RAGAS answer relevancy: generate n questions the ANSWER would answer, then return the
    mean cosine similarity between the original question and those generated ones. 0 if the
    answer is noncommittal/evasive. One LLM call + (batched) embeddings."""
    if cache is None:
        cache = {}
    if not answer:
        return 0.0
    gen = _question_generator.invoke(
        f"Generate {n} distinct questions that the ANSWER below would correctly and fully "
        "answer. Set noncommittal=true if the answer is evasive or admits it doesn't know.\n"
        f"ANSWER: {answer}"
    )
    if gen.noncommittal or not gen.questions:
        return 0.0
    vecs = _embed_many_cached([question] + list(gen.questions), cache)
    q_vec, gen_vecs = vecs[0], vecs[1:]
    return sum(cosine_similarity(q_vec, gv) for gv in gen_vecs) / len(gen_vecs)


def faithfulness_evaluation(validation_set, eval_results, file_path):
    """Average faithfulness over questions that retrieved chunk context (table-only
    questions have no chunk context to check, so they're skipped). Returns
    {"faithfulness", "n_questions", "per_item": {i: score}}."""
    child_to_parent, child_text, parent_text = _load_context_maps(file_path)
    total, scored, per_item = 0.0, 0, {}
    for i in sorted(validation_set.keys() & eval_results.keys()):
        context = _result_context(eval_results[i], child_to_parent, parent_text, child_text)
        if not context:
            continue
        score = faithfulness(validation_set[i].get("question", ""),
                             eval_results[i].get("response", ""), context)
        if score is None:
            continue
        per_item[i] = round(score, 4)
        total += score
        scored += 1
    avg = round(total / scored, 4) if scored else 0.0
    return {"faithfulness": avg, "n_questions": scored, "per_item": per_item}


def answer_relevancy_evaluation(validation_set, eval_results, file_path, n=3):
    """Average answer relevancy over every aligned question. Reuses the per-file embedding
    cache. Returns {"answer_relevancy", "n_questions", "per_item": {i: score}}."""
    cache_path = _embeddings_cache_path(file_path)
    cache = _load_cache(cache_path)
    n_before = len(cache)
    total, scored, per_item = 0.0, 0, {}
    for i in sorted(validation_set.keys() & eval_results.keys()):
        answer = eval_results[i].get("response", "")
        if not answer:
            continue
        score = answer_relevancy(validation_set[i].get("question", ""), answer, n, cache)
        per_item[i] = round(score, 4)
        total += score
        scored += 1
    if len(cache) > n_before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    avg = round(total / scored, 4) if scored else 0.0
    return {"answer_relevancy": avg, "n_questions": scored, "per_item": per_item}


def _format_llm_metrics(data):
    """Readable text block from an llm-metrics dict (freshly computed or loaded cache)."""
    faith, relev = data["faithfulness"], data["answer_relevancy"]
    questions = data.get("questions", {})
    lines = [f"LLM-judge metrics on {data.get('results_file', '?')} "
             f"({faith.get('n_questions', 0)} questions)",
             f"\n  Faithfulness     (avg): {faith.get('faithfulness')}",
             f"  Answer Relevancy (avg): {relev.get('answer_relevancy')}",
             "\n  Per question:"]
    f_items, r_items = faith.get("per_item", {}), relev.get("per_item", {})
    for k in sorted(set(f_items) | set(r_items), key=int):
        lines.append(f"    Q{k}  faithfulness={f_items.get(k, '-')}  "
                     f"answer_relevancy={r_items.get(k, '-')}")
        q = questions.get(str(k))
        if q:
            lines.append(f"        {q[:75]}")
    return "\n".join(lines)


def run_llm_metrics(file_path, results_name="eval_results.json", n=3):
    """Faithfulness + answer relevancy for `results_name`, CACHED to a json so re-runs make
    ZERO API calls (the LLM-judge calls only happen the first time). Also writes a readable
    txt. Cache files: processed/<hash>/<stem>_llm_metrics.{json,txt} where stem is derived
    from results_name (eval_results.json -> eval, eval_results_small.json -> eval_small).
    Delete the json to force a recompute (e.g. after regenerating eval_results)."""
    base = os.path.join(PROCESSED_DIR, file_hash(file_path))
    stem = results_name.rsplit(".", 1)[0].replace("eval_results", "eval")
    json_path = os.path.join(base, f"{stem}_llm_metrics.json")
    txt_path = os.path.join(base, f"{stem}_llm_metrics.txt")

    if os.path.exists(json_path):                      # already computed -> no API calls
        print(f"{stem}_llm_metrics.json present -> loading (no API calls)")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        print(_format_llm_metrics(data))
        return data

    ds, res = load_eval_data(file_path, results_name=results_name)
    faith = faithfulness_evaluation(ds, res, file_path)
    relev = answer_relevancy_evaluation(ds, res, file_path, n=n)
    # json object keys must be strings; normalize per_item so reload matches compute
    faith["per_item"] = {str(k): v for k, v in faith["per_item"].items()}
    relev["per_item"] = {str(k): v for k, v in relev["per_item"].items()}
    data = {
        "document": file_path,
        "results_file": results_name,
        "faithfulness": faith,
        "answer_relevancy": relev,
        "questions": {str(i): ds[i]["question"] for i in ds},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    text = _format_llm_metrics(data)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nSaved -> {json_path}\nSaved -> {txt_path}")
    return data


if __name__ == "__main__":
    file_name = str(DATA_DIR / "biology.pdf")
    eval_dataset, eval_results = load_eval_data(file_name)

    # collect everything into one readable report (printed AND saved to disk)
    report = []
    def out(s=""):
        print(s)
        report.append(s)

    retrieval_scores = retrieval_evaluation(eval_dataset, eval_results)
    semantic_scores = semantic_evaluation(eval_dataset, eval_results, file_name)
    keyword_exact = keyword_evaluation(eval_dataset, eval_results)
    keyword_fuzzy = keyword_evaluation(eval_dataset, eval_results, fuzzy=True,
                                       file_path=file_name)

    out(f"Document: {file_name}")
    out(f"Questions evaluated: {len(eval_dataset)}")
    out("\n================== METRICS ==================")
    out(f"  Retrieval        : {retrieval_scores}")
    out(f"  Semantic         : {semantic_scores}")
    out(f"  Keyword (exact)  : {keyword_exact}")
    out(f"  Keyword (fuzzy)  : {keyword_fuzzy}")

    # keyword failures with ground-truth + LLM answer (this prints on its own too)
    report.append(print_keyword_failures(eval_dataset, eval_results, file_path=file_name))

    report_path = os.path.join(PROCESSED_DIR, file_hash(file_name), "eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nSaved report -> {report_path}")

    # LLM-judge metrics (faithfulness + answer relevancy) for the full set.
    # CACHED: the first run computes + saves; every run after loads with NO API calls.
    print("\n========== LLM-JUDGE METRICS ==========")
    run_llm_metrics(file_name)