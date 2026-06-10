"""Agent assembly: build_agent wires a processed document into a ready-to-use LangChain
agent (retrievers + tools + model), and build_retrievers exposes the same raw retrievers
for retrieval ablations."""

import os

from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from config import (
    api_key as DEFAULT_OPENAI_KEY,
    CHAT_MODEL,
    EMBED_MODEL,
    AGENT_TEMPERATURE,
    MAX_COMPLETION_TOKENS,
    SYSTEM_PROMPT,
    PROCESSED_DIR,
    file_hash,
)
from enrich import clearance_level
from embeddings import CachingEmbeddings
from pipeline import process_document
from retrievers import build_hybrid_retriever
from tools import make_retrieve_context, make_table_tools
from eval_dataset import generate_eval_dataset


class ResponseFormat(BaseModel):
    message: str


def build_agent(file_path, *, generate_eval=False, role="employee", api_key=None,
                _exports=None, query_cache_path=None):
    """Process a file (pdf/docx/pptx, with caching) and return a ready-to-use agent.

    Returns (agent, retrieval_log, reset_turn). retrieval_log is shared by reference with
    the retrieve_context tool so a caller (e.g. the Streamlit UI) can read each turn's
    scores; reset_turn() MUST be called before every agent.invoke().

    api_key: the OpenAI key to use for embeddings + the LLM. If None, falls back to the
    .env key (config.api_key) — so the Streamlit UI can pass a user-supplied key while the
    CLI / eval scripts keep using the developer's .env.

    _exports: optional dict — if given, the raw retrievers (bm25/semantic/hybrid) and
    chunks are stashed into it before returning, WITHOUT changing the public return
    tuple. build_retrievers() uses this so a retrieval ablation tests the exact same
    objects the agent uses (one source of truth for k / weights / fusion)."""
    # RBAC: the caller's role decides the clearance the retriever enforces.
    clearance = clearance_level(role)
    print(f"Agent role: {role} (clearance level {clearance})")

    # use the caller-supplied key (e.g. entered in the UI), else the .env default
    key = api_key or DEFAULT_OPENAI_KEY

    embeddings = CachingEmbeddings(
        OpenAIEmbeddings(model=EMBED_MODEL, api_key=key),
        cache_path=query_cache_path,
    )

    doc = process_document(file_path, embeddings, api_key=key)

    if generate_eval:
        if not os.path.exists(doc.eval_path):
            generate_eval_dataset(doc.eval_path, doc.chunk_records, doc.table_metadata,
                                  doc.db, api_key=key)
        else:
            print(f"Eval dataset already exists at {doc.eval_path}")

    retrievers = build_hybrid_retriever(doc.chunks, doc.vectorstore)

    # each retrieve_context call appends one record here so a UI (Streamlit)
    # can show the same scores we print to the console.
    retrieval_log = []

    # per-turn guard so the agent can't loop on retrieve_context; reset_turn()
    # (returned below) must be called once before each agent.invoke().
    turn_state = {"calls": 0, "last_serialized": "", "last_docs": []}

    retrieve_context = make_retrieve_context(
        hybrid_retriever=retrievers["hybrid_retriever"],
        bm25_debugger=retrievers["bm25_debugger"],
        vectorstore=doc.vectorstore,
        chunks=doc.chunks,
        parents_by_id=doc.parents_by_id,
        clearance=clearance,
        role=role,
        doc_hash=doc.doc_hash,
        retrieval_log=retrieval_log,
        turn_state=turn_state,
    )
    get_table_metadata, query_pdf_tables = make_table_tools(doc.db, doc.table_metadata)

    model = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=AGENT_TEMPERATURE,
        api_key=key,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    tools = [
        retrieve_context,
        query_pdf_tables,
        get_table_metadata,
    ]

    agent = create_agent(
        model=model,
        tools=tools,
        response_format=ToolStrategy(ResponseFormat),
        system_prompt=SYSTEM_PROMPT,
    )

    def reset_turn():
        """Reset the per-turn retrieve_context guard. Call before each invoke."""
        turn_state["calls"] = 0

    # optional: hand back the raw retrievers (for ablations) without changing the
    # public return signature — see build_retrievers() below.
    if _exports is not None:
        _exports.update({
            "bm25_retriever": retrievers["bm25_retriever"],
            "semantic_retriever": retrievers["semantic_retriever"],
            "hybrid_retriever": retrievers["hybrid_retriever"],
            "chunks": doc.chunks,
        })

    return agent, retrieval_log, reset_turn


def build_retrievers(file_path, role="employee", query_cache_path=None, api_key=None):
    """Build (and return) the SAME three retrievers the agent uses, for retrieval
    ablations: {"bm25_retriever", "semantic_retriever", "hybrid_retriever", "chunks"}.

    Reuses build_agent so chunking, embeddings, k, and the RRF fusion weights stay a
    single source of truth (no drift from the live pipeline). No agent.invoke() / LLM
    calls happen here — only the cached index is loaded and the retrievers constructed.
    role is accepted for symmetry but does NOT affect raw retriever output (RBAC
    redaction lives in retrieve_context, not in the retrievers themselves).

    Query embeddings are persisted to processed/<hash>/query_embeddings.json by default,
    so the first ablation run embeds each question once and every later run makes ZERO
    embedding calls. Pass query_cache_path="" to disable persistence.
    """
    if query_cache_path is None:
        query_cache_path = os.path.join(
            PROCESSED_DIR, file_hash(file_path), "query_embeddings.json"
        )
    exports = {}
    build_agent(file_path, generate_eval=False, role=role, api_key=api_key,
                _exports=exports, query_cache_path=query_cache_path or None)
    return exports
