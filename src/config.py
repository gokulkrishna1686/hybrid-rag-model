"""Central configuration: paths, tunable constants, the agent system prompt, and a few
shared helpers. This module is imported (directly or transitively) by every entry point,
so its import-time side effects — forcing UTF-8 console output and loading the .env file —
apply everywhere.
"""

import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows consoles default to cp1252 and crash when printing non-ASCII characters
# (e.g. the currency symbols in the eval data). Force UTF-8 so prints never blow up.
# Every entry point imports this module, so this covers them all.
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


# ---------------------------------------------------------------------------
# Parent-child chunking (chars)
# ---------------------------------------------------------------------------
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
CHAT_MODEL = "gpt-4.1-mini"          # the answering agent + eval-dataset generation
EMBED_MODEL = "text-embedding-3-small"
AGENT_TEMPERATURE = 0.2
MAX_COMPLETION_TOKENS = 1000
EVAL_TEMPERATURE = 0.3               # eval-dataset question generation

# ---------------------------------------------------------------------------
# Retrieval + hybrid fusion
# ---------------------------------------------------------------------------
RETRIEVE_K = 3                       # top-k per retriever (BM25 and dense)
RRF_C = 60                           # Reciprocal Rank Fusion constant
SEMANTIC_WEIGHT = 0.6                # dense vs BM25 fusion weight
BM25_WEIGHT = 1 - SEMANTIC_WEIGHT    # 0.4


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
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
