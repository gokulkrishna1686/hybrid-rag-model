"""The agent's tools, built as factories that close over a processed document's retrievers,
parents, SQL db, and per-turn state:

  - retrieve_context : hybrid (BM25 + dense) retrieval with RRF logging, child->parent
                       expansion, optional entity filter, and RBAC redaction.
  - get_table_metadata / query_pdf_tables : schema inspection + SQL over the extracted tables.
"""

from langchain.tools import tool

from retrievers import retrieval_scores
from rbac import apply_clearance, serialize_docs


def expand_to_parents(retrieved_docs, parents_by_id):
    """Swap each matched child for its parent, de-duplicated by parent_id (preserving rank
    order). Returns (seen_parents, parent_docs)."""
    seen_parents = set()
    parent_docs = []
    for doc in retrieved_docs:
        pid = doc.metadata.get("parent_id")
        if pid in seen_parents:
            continue
        seen_parents.add(pid)
        parent_docs.append(parents_by_id.get(pid, doc))
    return seen_parents, parent_docs


def filter_by_mention(parent_docs, must_mention):
    """Self-query: keep only parents whose entity list mentions the requested term."""
    needle = must_mention.lower()
    return [
        p for p in parent_docs
        if any(needle in e.lower() for e in p.metadata.get("entities", []))
    ]


def make_table_tools(db, table_metadata):
    """Build the two SQL tools (get_table_metadata, query_pdf_tables) over `db`."""

    @tool
    def get_table_metadata():
        """
        Returns available SQL tables: CREATE TABLE schema (with column types)
        plus a few sample rows per table so the agent can pick the right table
        and write correct SQL.
        """
        page_index = "\n".join(
            f"-- {m['table_name']} (page {m['page']}): columns = {m['columns']}"
            for m in table_metadata
        )

        schema = db.get_table_info()

        payload = f"PAGE INDEX:\n{page_index}\n\nSCHEMA & SAMPLES:\n{schema}"

        print(f"METADATA RETRIEVED!")

        return payload

    @tool
    def query_pdf_tables(query: str):
        """
        Query structured PDF table data using SQL.
        """

        print("QUERY:", query)

        try:
            result = db.run(query)
            return str(result)

        except Exception as e:
            return f"SQL ERROR: {str(e)}"

    return get_table_metadata, query_pdf_tables


def make_retrieve_context(*, hybrid_retriever, bm25_debugger, vectorstore, chunks,
                          parents_by_id, clearance, role, doc_hash, retrieval_log,
                          turn_state):
    """Build the retrieve_context tool. It closes over the document's retrievers/parents,
    the caller's clearance, and the shared retrieval_log + per-turn guard state."""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str, must_mention: str = ""):
        """Retrieve information to help answer a query.

        Optionally set must_mention to a single name/term (e.g. a person or
        department) to only keep context that explicitly mentions it.
        """

        # hard cap: after the first call this turn, return the cached result
        # with a stop instruction so the agent can't loop on this tool.
        turn_state["calls"] += 1
        if turn_state["calls"] > 1:
            print(f"retrieve_context called {turn_state['calls']}x this turn -> returning cached result")
            stop_note = (
                "\n\n[NOTE: You already retrieved context this turn. Do NOT call "
                "retrieve_context again. Answer using the information above, or tell the "
                "user the information is unavailable or restricted for their access level.]"
            )
            return turn_state["last_serialized"] + stop_note, turn_state["last_docs"]

        top_bm25, semantic_results, rrf_scores = retrieval_scores(
            query, chunks, vectorstore, bm25_debugger
        )

        print(f"\n{'='*30}\nQUERY: {query}\n{'='*30}")
        print("\n===== BM25 RESULTS =====")
        for rank, (idx, score) in enumerate(top_bm25, 1):
            print(f"\nRank: {rank}\nBM25 Score: {score:.4f}\nchunk_id: {chunks[idx].metadata.get('chunk_id', '?')}")
        print("\n===== SEMANTIC RESULTS =====")
        for rank, (doc, score) in enumerate(semantic_results, 1):
            print(f"\nRank: {rank}\nSemantic Score: {score:.4f}\nchunk_id: {doc.metadata.get('chunk_id', '?')}")
        print("\n===== RRF RESULTS =====")
        for rank, (cid, _) in enumerate(sorted(rrf_scores.items(), key=lambda x: -x[1]), 1):
            print(f"Rank: {rank} | chunk_id: {cid}")

        retrieved_docs = hybrid_retriever.invoke(query)   # these are CHILDREN

        # swap each matched child back to its parent, dedupe by parent
        seen_parents, parent_docs = expand_to_parents(retrieved_docs, parents_by_id)

        retrieved_ids = [
            doc.metadata.get("chunk_id", "?") for doc in retrieved_docs
        ]
        print("CHILD MATCHES:", retrieved_ids)
        print("PARENT CONTEXT:", list(seen_parents))

        # --- self-query: keep only parents that mention the requested term ---
        if must_mention:
            parent_docs = filter_by_mention(parent_docs, must_mention)
            print(f"ENTITY FILTER '{must_mention}' -> {len(parent_docs)} parent(s) kept")

        # --- RBAC: enforce clearance; redact (don't drop) what's above it ---
        visible_docs, redacted_ids = apply_clearance(parent_docs, clearance, role)

        # record structured debug info for the UI (mirrors the prints above)
        retrieval_log.append({
            "query": query,
            "must_mention": must_mention,
            "bm25": [
                {"rank": r, "chunk_id": chunks[idx].metadata.get("chunk_id", "?"),
                 "score": round(float(score), 4)}
                for r, (idx, score) in enumerate(top_bm25, 1)
            ],
            "semantic": [
                {"rank": r, "chunk_id": doc.metadata.get("chunk_id", "?"),
                 "score": round(float(score), 4)}
                for r, (doc, score) in enumerate(semantic_results, 1)
            ],
            "rrf": [
                {"rank": r, "chunk_id": cid, "score": round(float(s), 6)}
                for r, (cid, s) in enumerate(sorted(rrf_scores.items(), key=lambda x: -x[1]), 1)
            ],
            "child_matches": retrieved_ids,
            "parents": list(seen_parents),
            "redacted": redacted_ids,
            # the source file's hash (one document per agent); empty when this turn
            # retrieved nothing (children/chunks are reported in child_matches).
            "sources": [doc_hash] if visible_docs else [],
        })

        serialized = serialize_docs(visible_docs)

        # cache for the per-turn guard (repeat calls reuse this)
        turn_state["last_serialized"] = serialized
        turn_state["last_docs"] = visible_docs

        return serialized, visible_docs

    return retrieve_context
