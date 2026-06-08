"""LLM generation of the evaluation dataset: a mixed set of keyword / conceptual /
paraphrased / table questions that stress all three retrieval paths, plus an optional
pass that appends extra rare-token keyword questions. Saved to
processed/<hash>/eval_dataset.json. Called by build_agent when generate_eval=True."""

import json
import os
import random
import re
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import api_key, CHAT_MODEL, EVAL_TEMPERATURE, PROCESSED_DIR, file_hash


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


def generate_eval_dataset(eval_path, chunk_records, table_metadata, db):
    print("Generating eval dataset (LLM)...")

    eval_llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=EVAL_TEMPERATURE,
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
        model=CHAT_MODEL,
        temperature=EVAL_TEMPERATURE,
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
