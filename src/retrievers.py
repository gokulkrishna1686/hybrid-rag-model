"""Builds the hybrid retrieval layer (BM25 + dense, fused with RRF) and computes the
per-query score breakdown used for retrieval-transparency logging."""

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from rank_bm25 import BM25Okapi

from config import RETRIEVE_K, RRF_C, BM25_WEIGHT, SEMANTIC_WEIGHT

# A slightly wider window than RETRIEVE_K, used ONLY for the score-breakdown logging so
# the transparency panel can show a few more candidates than the agent actually uses.
_DEBUG_TOPN = 4


def build_hybrid_retriever(chunks, vectorstore):
    """Construct the three retrievers the agent uses: BM25 (keyword), dense (semantic),
    and their RRF-fused ensemble. Returns a dict; bm25_debugger is a raw BM25Okapi over
    the same chunks, used to surface per-chunk BM25 scores for logging."""
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RETRIEVE_K
    bm25_debugger = BM25Okapi([doc.page_content.split() for doc in chunks])

    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVE_K},
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[BM25_WEIGHT, SEMANTIC_WEIGHT],
    )

    return {
        "bm25_retriever": bm25_retriever,
        "semantic_retriever": semantic_retriever,
        "hybrid_retriever": hybrid_retriever,
        "bm25_debugger": bm25_debugger,
    }


def retrieval_scores(query, chunks, vectorstore, bm25_debugger):
    """Compute the per-query score breakdown for logging. Returns
    (top_bm25, semantic_results, rrf_scores) where:
      - top_bm25         = [(chunk_index, bm25_score), ...]  (ranked)
      - semantic_results = [(doc, distance), ...]            (ranked; lower distance = closer)
      - rrf_scores       = {chunk_id: fused_score}
    """
    bm25_scores = bm25_debugger.get_scores(query.split())
    top_bm25 = sorted(enumerate(bm25_scores), key=lambda x: -x[1])[:_DEBUG_TOPN]
    semantic_results = vectorstore.similarity_search_with_score(query, k=_DEBUG_TOPN)

    rrf_scores = {}
    for rank, (idx, _) in enumerate(top_bm25, 1):
        cid = chunks[idx].metadata.get("chunk_id", f"?_{idx}")
        rrf_scores[cid] = rrf_scores.get(cid, 0) + BM25_WEIGHT / (RRF_C + rank)
    for rank, (doc, _) in enumerate(semantic_results, 1):
        cid = doc.metadata.get("chunk_id", "?")
        rrf_scores[cid] = rrf_scores.get(cid, 0) + SEMANTIC_WEIGHT / (RRF_C + rank)

    return top_bm25, semantic_results, rrf_scores
