from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import BaseModel
import os
import sys
import shutil
import hashlib
from pathlib import Path
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import json
import random
import re
from typing import Literal

from extract_images import (
    extract_and_caption_images,
    image_to_documents,
    descriptions_to_serializable,
    descriptions_from_serializable,
)
from extract_tables import extract_tables
from extract_text import load_text_docs
from enrich import enrich_text, redact, SENSITIVITY_ORDER, clearance_level

# Windows consoles default to cp1252 and crash when printing non-ASCII characters
# (e.g. the currency symbols in the eval data). Force UTF-8 so prints never blow up.
# Every entry point (main / eval / test_eval) imports this module, so this covers them.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Anchor everything to the repo root (this file lives in src/), so data, cache, AND the
# .env file resolve to the SAME place no matter which directory a script is launched
# from. This is why `python src/main.py` (from root) and `python main.py` (from inside
# src/) both use the one data_files/, processed/, and .env at the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_files"
PROCESSED_DIR = BASE_DIR / "processed"

load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("OPENAI_API_KEY")


class ResponseFormat(BaseModel):
    message: str


prompt = """
You answer questions about uploaded company reports.

Tool usage rules:

1. get_table_metadata()
- Returns a page index, CREATE TABLE schemas (with column types), and sample rows per table.
- ALWAYS use this before generating SQL queries.
- Pick the table whose columns and sample rows match the question.

2. query_pdf_tables(query: str)
- Use for structured table data.
- Use for calculations, averages, filtering, counts, rankings, IDs, salaries, budgets, and exact lookups.
- Input must be a valid SQLite query.
- Trust the column types from the schema (numeric columns are REAL/INTEGER — no need to CAST or strip commas).

3. retrieve_context(query: str, must_mention: str = "")
- Use ONLY for semantic document retrieval, summaries, explanations, charts, images, and general report context.
- DO NOT use this for exact table calculations or SQL-style queries.
- If the question is clearly about ONE specific person, department, or named
  entity, pass that name as must_mention to narrow the results. Otherwise leave
  it empty.
- Call AT MOST ONCE per user question. Pack everything you need to look up into a
  single rich query (combine keywords, synonyms, and the concepts you're after).
  Do NOT call it again hoping for different results.

General tool rules:
- Do not call the same tool more than once per question. If the first call did not
  return exactly what you wanted, answer with what you have instead of retrying.
- get_table_metadata only needs to be called once per question — cache the result
  mentally and reuse it for any follow-up SQL in the same turn.

For table/numerical questions:
1. inspect schema
2. generate SQL
3. execute SQL

Respond in under 30 words.
"""


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def split_parent_child(
    docs,
    parent_chunk_size=2000,
    parent_chunk_overlap=200,
    child_chunk_size=400,
    child_chunk_overlap=50,
):
    """
    Two-level split:
      - parents: big chunks that carry enough context to answer with.
      - children: small chunks we actually embed + run BM25 on (precise matching).
    Each child stores the parent_id it came from so we can swap it back at
    retrieval time.

    Returns (children, parents_by_id).
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        add_start_index=True,
    )

    parents = parent_splitter.split_documents(docs)

    children = []
    parents_by_id = {}

    for p_idx, parent in enumerate(parents):
        parent_id = f"parent_{p_idx}"
        parent.metadata["parent_id"] = parent_id
        parents_by_id[parent_id] = parent

        # split_documents copies the parent's metadata onto each child,
        # so parent_id (and source/page) ride along automatically.
        children.extend(child_splitter.split_documents([parent]))

    return children, parents_by_id


class _EvalItem(BaseModel):
    question: str
    ground_truth: str
    answer_type: Literal["literal", "descriptive"]
    keywords: list[str] = []
    expected_chunk_ids: list[str] = []
    expected_table_names: list[str] = []


class _EvalDataset(BaseModel):
    items: list[_EvalItem]


def _sanitize_eval_records(items, valid_chunk_ids, valid_table_names):
    """Turn raw LLM _EvalItems into clean eval_dataset records: recover garbled ids
    (e.g. "chunk_7'}]"), drop references that don't actually exist, and keep keywords
    only for descriptive answers. Shared by the full generator and the keyword pass."""
    def clean_chunk_id(raw):
        match = re.search(r"chunk_\d+", raw)
        return match.group(0) if match else None

    def build_contexts(item):
        contexts = []
        seen = set()
        for raw in item.expected_chunk_ids:
            cid = clean_chunk_id(raw)
            if cid in valid_chunk_ids and cid not in seen:
                seen.add(cid)
                contexts.append({"chunk_id": cid})
        for name in item.expected_table_names:
            name = name.strip()
            if name in valid_table_names and name not in seen:
                seen.add(name)
                contexts.append({"table_name": name})
        return contexts

    return [
        {
            "question": item.question,
            "ground_truth": item.ground_truth,
            "answer_type": item.answer_type,
            "keywords": (
                [k.strip() for k in item.keywords if k.strip()]
                if item.answer_type == "descriptive" else []
            ),
            "expected_contexts": build_contexts(item),
        }
        for item in items
    ]


def _generate_eval_dataset(eval_path, chunk_records, table_metadata, db):
    print("Generating eval dataset (LLM)...")

    eval_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3,
        api_key=api_key,
    ).with_structured_output(_EvalDataset)

    random.seed(42)

    def eval_chunks_filter(chunk_records, entity_keep_ratio=0.5):
        """Keep all text + summaries; sample entity chunks per image."""
        entities_by_image = {}
        for rec in chunk_records:
            if rec.get("type") == "image_entity":
                img_id = rec.get("image_id")
                entities_by_image.setdefault(img_id, []).append(rec["chunk_id"])

        kept_entity_ids = set()
        for img_id, cids in entities_by_image.items():
            n_keep = max(1, round(len(cids) * entity_keep_ratio))
            kept_entity_ids.update(random.sample(cids, n_keep))

        return [
            rec for rec in chunk_records
            if rec.get("type") != "image_entity" or rec["chunk_id"] in kept_entity_ids
        ]

    eval_chunks = eval_chunks_filter(chunk_records)

    chunks_context = "\n\n".join(
        f"[{rec['chunk_id']}] {rec['text']}" for rec in eval_chunks
    )

    tables_context = db.get_table_info() if table_metadata else ""

    num_chunk_questions = max(1, round(len(eval_chunks) * 0.4))
    num_table_questions = max(1, len(table_metadata)) if table_metadata else 0
    num_questions = num_chunk_questions + num_table_questions

    # descriptive (explanatory) answers can only come from chunk questions — table
    # questions are naturally literal — so guarantee a healthy share of them, else
    # the set ends up dominated by single-value lookups.
    num_descriptive_questions = min(
        num_chunk_questions, max(3, round(num_chunk_questions * 0.5))
    )
    num_literal_chunk_questions = num_chunk_questions - num_descriptive_questions

    eval_prompt = f"""You are generating an evaluation dataset for a hybrid RAG system
that combines BM25 (keyword) retrieval, dense semantic retrieval, and a SQL agent
that queries structured tables extracted from the PDF. The eval set must stress all
three paths.

Generate exactly {num_questions} evaluation questions, split into these categories
(these counts are a HARD requirement — match them):
- {num_descriptive_questions} DESCRIPTIVE chunk questions — answer_type="descriptive".
  Explanatory "why / how / describe / explain / summarize" questions whose ground_truth is
  a 1-2 sentence explanation, each with a non-empty keywords list. Answerable from the
  CHUNKS (prose + image captions).
- {num_literal_chunk_questions} LITERAL chunk questions — answer_type="literal".
  Single-value factual lookups (a name, date, number, or short phrase) from the CHUNKS.
- {num_table_questions} TABLE questions — answer_type="literal".
  SQL lookups, aggregations, filters, and rankings answerable from the TABLES.
Do NOT let single-value literal lookups dominate — honour the descriptive count above.

For each item provide:
- question: a clear, specific question answerable strictly from the provided material
- ground_truth: the concise correct answer (a value, name, phrase, or 1-2 sentence explanation)
- answer_type: "literal" if the answer is a single exact value (a number like 8.8, a
  boolean like True, a date, a name, an amount, or a short exact phrase); "descriptive"
  if the answer is an explanation or 1-2 sentence summary.
- keywords: for a "descriptive" answer, the essential terms or short phrases that MUST
  appear in a correct answer (e.g. ["AI automation", "customer support"]), used to score
  by keyword coverage. For a "literal" answer use an EMPTY list — the ground_truth is the
  exact value and is matched literally.
- expected_chunk_ids: a list of chunk_id strings (e.g. ["chunk_7"]) whose text is
  ACTUALLY needed to answer. Copy the exact id from inside the [square brackets] in the
  Chunks section below. Use an empty list if the answer does not come from any chunk.
- expected_table_names: a list of table_name strings (e.g. ["page_2_table_0"]) that must
  be queried to answer. Copy the exact name from the Tables schema below. Use an empty
  list if the answer does not require a table.
Include ONLY the contexts genuinely used to derive the answer. A chunk/prose question
must have an EMPTY expected_table_names, and a table/numeric question must have an EMPTY
expected_chunk_ids — do not attach a table that is not actually queried, nor a chunk that
is not actually read. List both only in the rare case the answer truly needs both.
Each list item must be a single bare id string only — never embed JSON, braces, quotes,
or punctuation inside it.

CHUNK question style:
- Vary the descriptive questions across why / how / what-does-X-mean / describe / summarize.
- The literal chunk lookups should reuse rare terms verbatim — good for BM25.
- Make ~30% of the chunk questions paraphrased / semantic: phrase them so the wording
  deliberately AVOIDS the chunk's exact keywords (synonyms, rephrasing, indirect framing)
  to stress the dense retriever. This applies to both descriptive and literal chunk questions.

TABLE question style (for the table questions):
- Mix exact lookups ("what is the salary of employee X"), aggregations
  ("average salary in department Y", "highest performer"), filters
  ("how many employees scored above 80"), and rankings.
- The ground_truth should be the actual answer computable from the table data,
  not the SQL query itself.
- Reference the correct table_name based on the schema below.

Other rules:
- Do NOT invent facts. Only ask about content actually present in the chunks/tables.
- Spread chunk questions across different chunks; spread table questions across tables.
- Image-caption chunks (descriptions of figures/charts) are fair game.

Chunks:
{chunks_context}

Tables (SQLite schema + sample rows):
{tables_context}
"""

    eval_data = eval_llm.invoke(eval_prompt)

    # recover garbled ids, keep only references that actually exist in the chunks/tables
    valid_chunk_ids = {rec["chunk_id"] for rec in eval_chunks}
    valid_table_names = (
        {m["table_name"] for m in table_metadata} if table_metadata else set()
    )
    eval_records = _sanitize_eval_records(
        eval_data.items, valid_chunk_ids, valid_table_names
    )

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=4, ensure_ascii=False)
    print(f"Eval dataset saved to {eval_path}")


def generate_keyword_questions(file_path, n=6):
    """Append N exact-token / keyword questions to eval_dataset.json — the case BM25 is
    meant to win: a rare LITERAL token (an ID, code, label, version, model number, proper
    noun) that dense embeddings blur but keyword search nails. The base set skews semantic,
    so these balance it and make a retrieval weight sweep trustworthy.

    Existing questions are preserved; duplicates (same question text) are skipped. Reads
    the cached chunks.json for the file's hash; eval_dataset.json must already exist.
    Returns the list of newly added records.
    """
    doc_hash = file_hash(file_path)
    processed_dir = str(PROCESSED_DIR / doc_hash)
    chunks_path = os.path.join(processed_dir, "chunks.json")
    eval_path = os.path.join(processed_dir, "eval_dataset.json")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    chunks_context = "\n\n".join(
        f"[{rec['chunk_id']}] {rec['text']}" for rec in chunk_records
    )
    valid_chunk_ids = {rec["chunk_id"] for rec in chunk_records}

    eval_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3,
        api_key=api_key,
    ).with_structured_output(_EvalDataset)

    prompt = f"""You are adding KEYWORD / EXACT-TOKEN questions to an evaluation set for a
hybrid RAG system. These specifically stress the BM25 (keyword) retriever, NOT the dense
semantic retriever.

Generate exactly {n} questions. EVERY question must hinge on a RARE, LITERAL token that
appears VERBATIM in only one (or very few) chunk(s) — an identifier, code, label, version
string, model/part number, room name, or other proper noun (e.g. "R-14", "UPS-Secondary",
"ISO 27001", "Orion Hall", "v3.2", "Dell PowerEdge").

Rules for EACH question:
- answer_type MUST be "literal". keywords MUST be an empty list.
- The question MUST contain the exact rare token, spelled exactly as it appears in the
  chunk, and must be answerable from that token's chunk alone.
- ground_truth is the exact value/answer (a short phrase, number, name, or code).
- Strongly PREFER opaque tokens (codes, IDs, labels, model/part numbers, version strings)
  over ordinary words — those are what dense embeddings struggle with and BM25 excels at.
- expected_chunk_ids: the single chunk_id whose text literally contains that token. Copy
  the exact id from inside the [square brackets] below. expected_table_names MUST be empty.
- Do NOT invent facts. Spread the questions across DIFFERENT chunks. No duplicates.

Chunks:
{chunks_context}
"""

    print(f"Generating {n} keyword/exact-token question(s) (LLM)...")
    eval_data = eval_llm.invoke(prompt)
    new_records = _sanitize_eval_records(eval_data.items, valid_chunk_ids, set())

    # defensive: keep only literal, chunk-grounded items (drop anything the model
    # mislabeled or that lost its chunk reference during sanitization)
    new_records = [
        r for r in new_records
        if r["answer_type"] == "literal"
        and any("chunk_id" in c for c in r["expected_contexts"])
    ]

    with open(eval_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_questions = {r["question"] for r in existing}
    added = [r for r in new_records if r["question"] not in existing_questions]

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(existing + added, f, indent=4, ensure_ascii=False)

    print(f"Added {len(added)} keyword question(s) -> {eval_path} "
          f"(was {len(existing)}, now {len(existing) + len(added)})")
    return added


class _CachingEmbeddings(Embeddings):
    """Wraps an embeddings backend and memoizes embed_query: the same query string is
    embedded once, then served from a dict. Harmless for the live agent (queries rarely
    repeat in a session) but a big saving for tools that re-issue the same queries — e.g.
    the ablation weight-sweep that re-runs every question across many fusion weights.

    If cache_path is given, the query cache is also PERSISTED to disk (loaded on init,
    written on each new query) — so a tool like the ablation embeds each question only the
    very first time it is ever run, and makes ZERO embedding calls on every run after that.
    embed_documents passes straight through (build-time work, no repeats)."""

    def __init__(self, base, cache_path=None):
        self._base = base
        self._cache_path = cache_path
        self._query_cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    self._query_cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._query_cache = {}

    def embed_documents(self, texts):
        return self._base.embed_documents(texts)

    def embed_query(self, text):
        if text not in self._query_cache:
            self._query_cache[text] = self._base.embed_query(text)
            if self._cache_path:                       # persist so re-runs are free
                os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
                with open(self._cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._query_cache, f)
        return self._query_cache[text]


def build_agent(file_path, *, generate_eval=False, role="employee", _exports=None,
                query_cache_path=None):
    """Process a file (pdf/docx/pptx, with caching) and return a ready-to-use agent.

    _exports: optional dict — if given, the raw retrievers (bm25/semantic/hybrid) and
    chunks are stashed into it before returning, WITHOUT changing the public return
    tuple. build_retrievers() uses this so a retrieval ablation tests the exact same
    objects the agent uses (one source of truth for k / weights / fusion)."""
    # RBAC: the caller's role decides the clearance the retriever enforces.
    clearance = clearance_level(role)
    print(f"Agent role: {role} (clearance level {clearance})")

    doc_hash = file_hash(file_path)
    processed_dir = str(PROCESSED_DIR / doc_hash)
    chunks_path = os.path.join(processed_dir, "chunks.json")
    parents_path = os.path.join(processed_dir, "parents.json")
    image_cache_path = os.path.join(processed_dir, "image_descriptions.json")
    eval_path = os.path.join(processed_dir, "eval_dataset.json")
    chroma_path = os.path.join(processed_dir, "chroma_db")
    db_path = os.path.join(processed_dir, "tables.db")
    table_meta_path = os.path.join(processed_dir, "table_metadata.json")
    os.makedirs(processed_dir, exist_ok=True)

    if os.path.exists(db_path) and os.path.exists(table_meta_path):
        print(f"Tables already extracted (hash={doc_hash[:8]}). Loading cached .db + metadata.")
        with open(table_meta_path, "r", encoding="utf-8") as f:
            table_metadata = json.load(f)
    else:
        print(f"Extracting tables (hash={doc_hash[:8]})...")
        result = extract_tables(file_path, db_path=db_path)
        table_metadata = result["table_metadata"]
        with open(table_meta_path, "w", encoding="utf-8") as f:
            json.dump(table_metadata, f, indent=4, ensure_ascii=False)

    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        sample_rows_in_table_info=0
    )

    embeddings = _CachingEmbeddings(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key),
        cache_path=query_cache_path,
    )

    if os.path.exists(chunks_path) and os.path.exists(chroma_path) and os.path.exists(parents_path):
        print(f"File already processed (hash={doc_hash[:8]}). Loading cached chunks + vectorstore.")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_records = json.load(f)
        with open(parents_path, "r", encoding="utf-8") as f:
            parent_records = json.load(f)

        chunks = [
            Document(
                page_content=rec["text"],
                metadata={
                    "source": rec["document"],
                    "page": rec["page"],
                    "start_index": rec["start_index"],
                    "chunk_id": rec["chunk_id"],
                    "parent_id": rec.get("parent_id"),
                }
            )
            for rec in chunk_records
        ]

        parents_by_id = {
            rec["parent_id"]: Document(
                page_content=rec["text"],
                metadata={
                    "source": rec["document"],
                    "page": rec["page"],
                    "parent_id": rec["parent_id"],
                    "sensitivity": rec.get("sensitivity", "public"),
                    "has_pii": rec.get("has_pii", False),
                    "pii_types": rec.get("pii_types", []),
                    "entities": rec.get("entities", []),
                }
            )
            for rec in parent_records
        }

        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
    else:
        print(f"Processing file (hash={doc_hash[:8]})...")
        docs = load_text_docs(file_path)

        chunks, parents_by_id = split_parent_child(
            docs,
            parent_chunk_size=2000,
            parent_chunk_overlap=200,
            child_chunk_size=400,
            child_chunk_overlap=50,
        )

        # captions are expensive (gpt-4o vision per image) and never change for
        # the same file, so cache them separately from the chunk/Chroma rebuild.
        images_dir = os.path.join(processed_dir, "images")
        if os.path.exists(image_cache_path):
            print(f"Images already captioned (hash={doc_hash[:8]}). Loading cached captions.")
            with open(image_cache_path, "r", encoding="utf-8") as f:
                image_descriptions = descriptions_from_serializable(json.load(f))
        else:
            image_descriptions = extract_and_caption_images(file_path, output_folder=images_dir)
            with open(image_cache_path, "w", encoding="utf-8") as f:
                json.dump(descriptions_to_serializable(image_descriptions), f,
                          indent=4, ensure_ascii=False)

        image_chunks = []
        for item in image_descriptions:
            image_chunks.extend(image_to_documents(item))

        # images have no textual parent — each image chunk is its own parent.
        # store a SEPARATE copy as the parent: enrichment adds list metadata
        # (entities/pii_types) to parents, and Chroma rejects list/empty-list
        # metadata. The child we embed must stay free of those fields.
        for img_idx, child in enumerate(image_chunks):
            parent_id = f"parent_img_{img_idx}"
            child.metadata["parent_id"] = parent_id
            parents_by_id[parent_id] = Document(
                page_content=child.page_content,
                metadata=dict(child.metadata),
            )

        chunks.extend(image_chunks)

        # enrich every parent: entities (spaCy) + PII/sensitivity (Presidio).
        # we tag PARENTS (not children) because parents are what the LLM sees,
        # so access control + filtering happen on that unit.
        print(f"Enriching {len(parents_by_id)} parents (entities + sensitivity)...")
        for parent in parents_by_id.values():
            parent.metadata.update(enrich_text(parent.page_content))

        chunk_records = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            chunk.metadata["chunk_id"] = chunk_id
            chunk_records.append({
                "chunk_id": chunk_id,
                "parent_id": chunk.metadata.get("parent_id"),
                "document": chunk.metadata.get('source'),
                "page": chunk.metadata.get('page'),
                "start_index": chunk.metadata.get('start_index'),
                "type": chunk.metadata.get('type'),
                "image_id": chunk.metadata.get('image_id'),
                "text": chunk.page_content
            })

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_records, f, indent=4, ensure_ascii=False)

        # persist parents so we can rebuild parents_by_id from cache
        parent_records = [
            {
                "parent_id": pid,
                "document": p.metadata.get("source"),
                "page": p.metadata.get("page"),
                "type": p.metadata.get("type"),
                "image_id": p.metadata.get("image_id"),
                "sensitivity": p.metadata.get("sensitivity"),
                "has_pii": p.metadata.get("has_pii"),
                "pii_types": p.metadata.get("pii_types"),
                "entities": p.metadata.get("entities"),
                "text": p.page_content,
            }
            for pid, p in parents_by_id.items()
        ]
        with open(parents_path, "w", encoding="utf-8") as f:
            json.dump(parent_records, f, indent=4, ensure_ascii=False)

        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)

        vectorstore = Chroma.from_documents(
            documents=chunks,           # children get embedded
            embedding=embeddings,
            persist_directory=chroma_path
        )

    if generate_eval:
        if not os.path.exists(eval_path):
            _generate_eval_dataset(eval_path, chunk_records, table_metadata, db)
        else:
            print(f"Eval dataset already exists at {eval_path}")

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3
    bm25_debugger = BM25Okapi([doc.page_content.split() for doc in chunks])

    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    semantic_weight = 0.6
    bm25_weight = 1 - semantic_weight

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            semantic_retriever
        ],
        weights=[bm25_weight, semantic_weight]
    )

    # each retrieve_context call appends one record here so a UI (Streamlit)
    # can show the same scores we print to the console.
    retrieval_log = []

    # per-turn guard so the agent can't loop on retrieve_context; reset_turn()
    # (returned below) must be called once before each agent.invoke().
    turn_state = {"calls": 0, "last_serialized": "", "last_docs": []}

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

        bm25_scores = bm25_debugger.get_scores(query.split())
        top_bm25 = sorted(enumerate(bm25_scores), key=lambda x: -x[1])[:4]
        semantic_results = vectorstore.similarity_search_with_score(query, k=4)

        print(f"\n{'='*30}\nQUERY: {query}\n{'='*30}")
        print("\n===== BM25 RESULTS =====")
        for rank, (idx, score) in enumerate(top_bm25, 1):
            print(f"\nRank: {rank}\nBM25 Score: {score:.4f}\nchunk_id: {chunks[idx].metadata.get('chunk_id', '?')}")
        print("\n===== SEMANTIC RESULTS =====")
        for rank, (doc, score) in enumerate(semantic_results, 1):
            print(f"\nRank: {rank}\nSemantic Score: {score:.4f}\nchunk_id: {doc.metadata.get('chunk_id', '?')}")

        k = 60
        rrf_scores = {}
        for rank, (idx, _) in enumerate(top_bm25, 1):
            cid = chunks[idx].metadata.get("chunk_id", f"?_{idx}")
            rrf_scores[cid] = rrf_scores.get(cid, 0) + bm25_weight / (k + rank)
        for rank, (doc, _) in enumerate(semantic_results, 1):
            cid = doc.metadata.get("chunk_id", "?")
            rrf_scores[cid] = rrf_scores.get(cid, 0) + semantic_weight / (k + rank)
        print("\n===== RRF RESULTS =====")
        for rank, (cid, _) in enumerate(sorted(rrf_scores.items(), key=lambda x: -x[1]), 1):
            print(f"Rank: {rank} | chunk_id: {cid}")

        retrieved_docs = hybrid_retriever.invoke(query)   # these are CHILDREN

        # swap each matched child back to its parent, dedupe by parent
        seen_parents = set()
        parent_docs = []
        for doc in retrieved_docs:
            pid = doc.metadata.get("parent_id")
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            parent_docs.append(parents_by_id.get(pid, doc))

        retrieved_ids = [
            doc.metadata.get("chunk_id", "?") for doc in retrieved_docs
        ]
        print("CHILD MATCHES:", retrieved_ids)
        print("PARENT CONTEXT:", list(seen_parents))

        # --- self-query: keep only parents that mention the requested term ---
        if must_mention:
            needle = must_mention.lower()
            parent_docs = [
                p for p in parent_docs
                if any(needle in e.lower() for e in p.metadata.get("entities", []))
            ]
            print(f"ENTITY FILTER '{must_mention}' -> {len(parent_docs)} parent(s) kept")

        # --- RBAC: enforce clearance; redact (don't drop) what's above it ---
        visible_docs = []
        redacted_ids = []
        for p in parent_docs:
            level = SENSITIVITY_ORDER.get(p.metadata.get("sensitivity", "public"), 0)
            if level <= clearance:
                visible_docs.append(p)
            else:
                # too sensitive for this role -> mask the PII, keep the prose,
                # and ANNOUNCE the redaction so the agent treats it as an access
                # block (and stops) rather than a retrieval miss (and retries).
                redacted = redact(p.page_content)
                marker = (
                    f"[RESTRICTED - this content is above your '{role}' access level "
                    f"and has been redacted. This is an access restriction, NOT a "
                    f"missing result, so do not retry. Tell the user they are not "
                    f"authorized to view it.]"
                )
                visible_docs.append(Document(
                    page_content=f"{marker}\n{redacted}",
                    metadata={**p.metadata, "redacted": True}
                ))
                redacted_ids.append(p.metadata.get("parent_id"))
                print(f"REDACTED {p.metadata.get('parent_id')} "
                      f"(sensitivity={p.metadata.get('sensitivity')} > clearance={clearance})")

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

        serialized = "\n\n".join(
            (
                f"Source: {doc.metadata}\n"
                f"Content: {doc.page_content}"
            )
            for doc in visible_docs
        )

        # cache for the per-turn guard (repeat calls reuse this)
        turn_state["last_serialized"] = serialized
        turn_state["last_docs"] = visible_docs

        return serialized, visible_docs

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,
        api_key=api_key,
        max_completion_tokens=1000
    )

    tools = [
        retrieve_context,
        query_pdf_tables,
        get_table_metadata
    ]

    agent = create_agent(
        model=model,
        tools=tools,
        response_format=ToolStrategy(ResponseFormat),
        system_prompt=prompt
    )

    def reset_turn():
        """Reset the per-turn retrieve_context guard. Call before each invoke."""
        turn_state["calls"] = 0

    # optional: hand back the raw retrievers (for ablations) without changing the
    # public return signature — see build_retrievers() below.
    if _exports is not None:
        _exports.update({
            "bm25_retriever": bm25_retriever,
            "semantic_retriever": semantic_retriever,
            "hybrid_retriever": hybrid_retriever,
            "chunks": chunks,
        })

    # retrieval_log is shared by reference with retrieve_context so a caller can
    # read each turn's scores. reset_turn() MUST be called before every invoke.
    return agent, retrieval_log, reset_turn


def build_retrievers(file_path, role="employee", query_cache_path=None):
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
    build_agent(file_path, generate_eval=False, role=role, _exports=exports,
                query_cache_path=query_cache_path or None)
    return exports


def run_cli(file_path, role="employee"):
    agent, _, reset_turn = build_agent(file_path, generate_eval=True, role=role)

    while True:
        user_query = input("\nYou: ")

        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Exiting...")
            break

        reset_turn()
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_query
                    }
                ]
            }
        )

        print("\nAssistant:", result["structured_response"].message)


if __name__ == "__main__":
    file_name = str(DATA_DIR / "Employee Performance.docx")
    # role controls what the retriever is allowed to surface:
    #   guest -> public, employee -> internal, manager -> confidential
    run_cli(file_name, role="manager")