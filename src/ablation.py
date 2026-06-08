"""Retrieval ablation: BM25-only vs dense-only vs hybrid, on the same eval questions.

Runs each of the three retrievers over the CHUNK questions from eval_dataset.json and
scores them with the SAME retrieval metrics used in test_eval.py (precision / recall /
hit_rate / mrr / ndcg). This isolates the retrieval layer — no agent, no LLM answer
generation — so you can see whether the hybrid actually beats each single path, and in
particular whether the dense/semantic path is pulling its weight.

Only CHUNK questions are scored (expected_contexts containing a chunk_id). TABLE
questions are answered by the SQL agent, not the retrievers, so including them would just
score 0 for all three and add noise.

The only API calls are cheap query embeddings for the dense + hybrid paths (one per
question each) — no completion/LLM calls. Run from the project root with the venv python:
    ./.venv/Scripts/python.exe src/ablation.py
"""

import json
import os

from langchain_classic.retrievers import EnsembleRetriever

from agent import build_retrievers
from config import file_hash, PROCESSED_DIR, DATA_DIR
from test_eval import retrieval_evaluation


def _retrieved_chunk_ids(retriever, query):
    """Child chunk_ids a retriever returns for one query, in rank order (deduped,
    order preserved). This mirrors what eval_results records as `chunks_retrieved`."""
    ids = []
    for doc in retriever.invoke(query):
        cid = doc.metadata.get("chunk_id")
        if cid and cid not in ids:
            ids.append(cid)
    return ids


def _chunk_questions(file_path):
    """eval_dataset items whose answer comes from chunks (>=1 chunk_id in
    expected_contexts), keyed by a shared index. Table-only questions are dropped."""
    dataset_path = os.path.join(PROCESSED_DIR, file_hash(file_path), "eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    chunk_items = [
        it for it in items
        if any(ctx.get("chunk_id") for ctx in it.get("expected_contexts", []))
    ]
    return {i: it for i, it in enumerate(chunk_items)}


def run_ablation(file_path, role="manager", ks=(1, 3, 5, None), retrievers=None):
    """Score BM25-only / dense-only / hybrid on the chunk questions, across cutoffs k.

    Returns (scores, report_text) where scores = {k: {path_name: metrics_dict}}. At k=1
    and k=3 every path is capped to the same depth (a fair, apples-to-apples ranking
    comparison). At k=5/all the hybrid may use its larger fused candidate set (bm25/dense
    only ever return their top 3). Pass `retrievers` (from build_retrievers) to reuse a
    shared retriever set — its query-embedding cache then makes repeated queries free.
    """
    validation_set = _chunk_questions(file_path)
    if retrievers is None:
        retrievers = build_retrievers(file_path, role=role)

    paths = {
        "bm25_only":     retrievers["bm25_retriever"],
        "semantic_only": retrievers["semantic_retriever"],
        "hybrid":        retrievers["hybrid_retriever"],
    }

    # run each retriever once per question; cache the ranked ids so the k-sweep is free
    retrieved = {name: {} for name in paths}
    for name, retr in paths.items():
        for i, item in validation_set.items():
            retrieved[name][i] = _retrieved_chunk_ids(retr, item["question"])

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out(f"Retrieval ablation over {len(validation_set)} chunk question(s) "
        f"(table-only questions excluded).")
    out("Higher is better for every metric. bm25/dense return their top 3; "
        "hybrid fuses both, so k>=5 lets it use a larger set.\n")

    metric_keys = ["precision", "recall", "hit_rate", "mrr", "ndcg"]
    col = 15
    all_scores = {}
    for k in ks:
        ktxt = "all" if k is None else str(k)
        header = f"{'@k=' + ktxt:<14}" + "".join(f"{m:>{col}}" for m in metric_keys)
        out(header)
        out("-" * len(header))
        all_scores[k] = {}
        for name in paths:
            eval_results = {
                i: {"chunks_retrieved": retrieved[name][i], "tables_queried": []}
                for i in validation_set
            }
            scores = retrieval_evaluation(validation_set, eval_results, k=k)
            all_scores[k][name] = scores
            out(f"{name:<14}" + "".join(f"{scores[m]:>{col}.4f}" for m in metric_keys))
        # who wins each metric at this k (ties -> first)
        winners = {
            m: max(paths, key=lambda n: all_scores[k][n][m]) for m in metric_keys
        }
        out(f"{'winner':<14}" + "".join(f"{winners[m]:>{col}}" for m in metric_keys))
        out()

    return all_scores, "\n".join(lines)


def run_weight_sweep(file_path, role="manager", k=3,
                     bm25_weights=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.6, 0.7, 0.8, 0.9, 1.0), retrievers=None):
    """Sweep the BM25-vs-semantic fusion weight and score the hybrid at each setting, to
    find the weighting that maximizes retrieval quality on this doc (vs the current guess
    of bm25=0.4 / semantic=0.6).

    For each row the EnsembleRetriever is rebuilt with weights=[w_bm25, 1-w_bm25] and the
    SAME bm25/semantic base retrievers — so this is the real fusion, just re-weighted.
    NOTE the endpoints are not identical to bm25_only/semantic_only in run_ablation: the
    ensemble always UNIONS both candidate sets, it only changes how they are RRF-weighted
    (so w_bm25=0.0 is "semantic-ranked over both sets", not "semantic alone").

    Returns (rows, report_text) where rows = {w_bm25: metrics_dict}. Pass `retrievers` to
    reuse a shared set — the same queries are re-issued for every weight, so a shared
    query-embedding cache makes the whole sweep cost only one embedding per question.
    """
    validation_set = _chunk_questions(file_path)
    if retrievers is None:
        retrievers = build_retrievers(file_path, role=role)
    bm25 = retrievers["bm25_retriever"]
    semantic = retrievers["semantic_retriever"]

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    metric_keys = ["precision", "recall", "hit_rate", "mrr", "ndcg"]
    col = 15
    out(f"Weight sweep over {len(validation_set)} chunk question(s) @ k={k} "
        f"(current pipeline = bm25 0.4 / sem 0.6).\n")
    header = f"{'bm25 / sem':<14}" + "".join(f"{m:>{col}}" for m in metric_keys)
    out(header)
    out("-" * len(header))

    rows = {}
    for w_bm25 in bm25_weights:
        w_sem = round(1 - w_bm25, 3)
        ensemble = EnsembleRetriever(
            retrievers=[bm25, semantic],
            weights=[w_bm25, w_sem],
        )
        eval_results = {
            i: {
                "chunks_retrieved": _retrieved_chunk_ids(ensemble, item["question"]),
                "tables_queried": [],
            }
            for i, item in validation_set.items()
        }
        scores = retrieval_evaluation(validation_set, eval_results, k=k)
        rows[w_bm25] = scores
        label = f"{w_bm25:.1f} / {w_sem:.1f}"
        out(f"{label:<14}" + "".join(f"{scores[m]:>{col}.4f}" for m in metric_keys))

    out("\nBest fusion weight per metric:")
    for m in metric_keys:
        best = max(rows, key=lambda w: rows[w][m])
        out(f"  {m:<10} -> bm25={best:.1f} / sem={1 - best:.1f}  (score {rows[best][m]:.4f})")
    return rows, "\n".join(lines)


def _stringify_keys(d):
    """JSON object keys must be strings; our score dicts are keyed by k (incl. None) and
    by float weights. Coerce the top-level keys so json.dump is lossless and readable."""
    out = {}
    for key, val in d.items():
        out["all" if key is None else str(key)] = val
    return out


if __name__ == "__main__":
    file_name = str(DATA_DIR / "Employee Performance.docx")

    # build the retrievers ONCE and share them, so the per-process query-embedding cache
    # is shared across both runs -> each distinct question is embedded only one time.
    retrievers = build_retrievers(file_name, role="manager")

    abl_scores, abl_text = run_ablation(file_name, retrievers=retrievers)
    print()
    sweep_scores, sweep_text = run_weight_sweep(file_name, k=3, retrievers=retrievers)

    # --- save the results so they can be reviewed later -------------------------------
    out_dir = os.path.join(PROCESSED_DIR, file_hash(file_name))
    report_path = os.path.join(out_dir, "ablation_results.txt")
    json_path = os.path.join(out_dir, "ablation_results.json")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Document: {file_name}\n\n")
        f.write("=== RETRIEVAL ABLATION ===\n")
        f.write(abl_text + "\n\n")
        f.write("=== WEIGHT SWEEP ===\n")
        f.write(sweep_text + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "document": file_name,
                "ablation": _stringify_keys(abl_scores),
                "weight_sweep": _stringify_keys(sweep_scores),
            },
            f, indent=2,
        )

    print(f"\nSaved results -> {report_path}")
    print(f"Saved results -> {json_path}")
