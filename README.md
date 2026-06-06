# Hybrid RAG Model

A retrieval-augmented question-answering system for company reports (**PDF, DOCX, PPTX**). It combines **three retrieval paths** behind a single LangChain agent, and adds **chunk-level governance** — named-entity extraction and PII-based sensitivity labels that drive **role-based access control (RBAC)** over what each user is allowed to see.

The three retrieval paths:

1. **Keyword retrieval (BM25)** — exact-term and rare-keyword matches.
2. **Dense semantic retrieval (Chroma + OpenAI embeddings)** — paraphrased / conceptual questions.
3. **SQL over extracted tables** — calculations, aggregations, filters, and exact lookups on structured data pulled out of the document.

Text, image captions, and tables are all extracted from the uploaded file, and the agent decides which path to use per question.

## How it works

When a file is processed (`build_agent` in `main.py`), the pipeline:

1. **Loads text** (`extract_text.py`) per file type — `PyPDFLoader` for PDF, `python-docx` for DOCX, `python-pptx` for PPTX (one document per slide).
2. **Parent–child chunking** (`split_parent_child`): text is split into large **parent** chunks (2000 chars / 200 overlap) and small **child** chunks (400 / 50). Children are what get embedded and BM25-indexed (precise matching); at query time each matched child is swapped back to its **parent** before being handed to the LLM (richer context). See [Parent–child retrieval](#parentchild-retrieval).
3. **Extracts & captions images** (`extract_images.py`) using PyMuPDF (`fitz`) / zip media for DOCX·PPTX, and GPT-4o vision to produce a structured caption (type, summary, per-entity colors/shapes/text, keywords). Each image becomes one *summary* document plus one document per detected *entity*. Captions are cached (see [Caching](#caching)).
4. **Extracts tables** (`extract_tables.py`) with `pdfplumber`, cleans columns/cells, auto-detects numeric columns, and writes one SQLite table per detected table (`page_<n>_table_<i>`).
5. **Enriches chunks** (`enrich.py`) — see [Access control](#access-control-rbac).
6. **Builds retrievers**: a `BM25Retriever` and a Chroma vector store (`text-embedding-3-small`) combined in an `EnsembleRetriever` (BM25 weight `0.4`, semantic weight `0.6`, fused with Reciprocal Rank Fusion).
7. **Creates the agent** (`create_agent`, model `gpt-4.1-mini`) exposing three tools:
   - `retrieve_context(query, must_mention="")` — hybrid BM25 + semantic retrieval for prose, summaries, and image content. Optional `must_mention` narrows results to chunks that mention a given entity.
   - `get_table_metadata()` — returns table schemas and sample rows so the agent can pick the right table.
   - `query_pdf_tables(query)` — runs a SQLite query against the extracted tables.

### Parent–child retrieval

Small chunks match precisely but lack context; large chunks give context but match imprecisely. Parent–child gets both: embed/search the **children**, but return their **parents** to the LLM. Each child stores its `parent_id`; after the hybrid retriever returns children, they are de-duplicated up to their parents and only the parent text is serialized into the prompt. Image chunks are their own parent.

### Access control (RBAC)

Each **parent** chunk is enriched (`enrich.py`) with two local, offline models:

- **spaCy** (`en_core_web_sm`) → named entities (people, orgs, dates, money…). Used for the `must_mention` filter ("only chunks that mention X").
- **Presidio** → PII detection (emails, phones, SSNs, names…), mapped to a **3-tier sensitivity label**:

| Tier | Triggered by | Role cleared to see it |
|------|--------------|------------------------|
| `public` | no PII | `guest` (and above) |
| `internal` | names / locations (`PERSON`, `LOCATION`, `NRP`) | `employee` (and above) |
| `confidential` | strong PII (`EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`, …) | `manager` |

A role maps to a numeric **clearance** (`ROLE_CLEARANCE`): `guest → 0`, `employee → 1`, `manager → 2`. At retrieval time `retrieve_context` enforces `chunk_sensitivity ≤ clearance`:

- **At or below clearance** → returned as-is.
- **Above clearance** → **redacted, not dropped**: Presidio's anonymizer masks the PII (`<PERSON>`, `<EMAIL_ADDRESS>`…) and the text is prefixed with a `[RESTRICTED …]` marker so the agent treats it as an access block (and tells the user they're not authorized) rather than a retrieval miss.

A **per-turn guard** prevents the agent from looping on `retrieve_context` — only the first call per turn runs; further calls return the cached result with a stop instruction.

### Retrieval transparency

Every `retrieve_context` call records the **BM25 scores, semantic distances, RRF re-ranking, matched children, returned parents, and any redactions**. The Streamlit app shows these live in a right-hand panel so you can see exactly why each answer was retrieved.

### Caching

Processing is cached per file by MD5 hash under `processed/<hash>/`:

```
chunks.json              # child chunks (+ parent_id)
parents.json             # parent chunks (+ sensitivity, entities, pii_types)
image_descriptions.json  # cached GPT-4o captions (skips re-captioning)
chroma_db/               # embedded child vectors
tables.db                # extracted tables (SQLite)
table_metadata.json      # table schema/page index
eval_dataset.json        # optional, see below
```

Each file is fully isolated in its own hash folder, and re-running on the same file reuses the cache instead of re-extracting / re-embedding / re-captioning.

### Evaluation dataset (optional)

When `generate_eval=True`, an LLM generates a mixed evaluation set — keyword, conceptual, paraphrased, and table questions — saved to `processed/<hash>/eval_dataset.json` to stress all three retrieval paths.

## Project structure

```
src/                  # all application + eval code (run scripts from the project root)
  main.py             # Pipeline + agent (build_agent), RBAC enforcement, CLI entry point
  enrich.py           # spaCy NER + Presidio PII -> sensitivity tiers + role→clearance map
  extract_text.py     # Multi-format text loading (PDF / DOCX / PPTX)
  extract_images.py   # Image extraction (PyMuPDF / zip) + GPT-4o vision captioning
  extract_tables.py   # Table extraction (pdfplumber) into SQLite
  streamlit_app.py    # Streamlit chat UI: upload, role selector, retrieval-scores panel
  eval.py             # Eval runner (batch/interactive) -> processed/<hash>/eval_results.json
  test_eval.py        # Loads eval_dataset.json + eval_results.json as dicts for scoring
notebooks/            # Exploratory notebooks (BM25/Chroma, table extraction)
scratch/              # Scratch experiments, not used by the app
docs/                 # Project notes (TODO.md)
requirements.txt      # Pinned dependencies
processed/            # Per-file cache (gitignored)
data_files/           # Local documents (gitignored)
```

## Setup

Requires Python 3.11 and an OpenAI API key.

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# NER / PII stack (not pinned in requirements.txt) + the spaCy model
pip install spacy presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_sm
```

Create a `.env` file in the project root (see `.env.example`):

```
OPENAI_API_KEY=sk-...

# optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=hybrid-rag-model
```

LangChain/LangGraph pick up the `LANGSMITH_*` variables automatically — no code changes needed; each query shows up as a trace in LangSmith.

## Usage

> `data_files/` and `processed/` are anchored to the project root in code, so these
> commands work no matter which directory you launch them from. The examples assume
> you're at the project root.

### Streamlit app

```powershell
streamlit run src/streamlit_app.py
```

Upload a PDF/DOCX/PPTX in the sidebar, pick an **access level** (`guest` / `employee` / `manager`), wait for processing, then ask questions. The right-hand panel shows the retrieval scores and which parents were redacted for the current role. Changing the role rebuilds the agent with the new clearance.

> Try `data_files/Tier Demo.docx` (a doc with one section per sensitivity tier): ask the same question as `guest` vs `manager` to watch redaction kick in.

### Command line

```powershell
python src/main.py
```

By default it loads `data_files/Employee Performance.docx` as `role="employee"` — edit the `file_name` / `role` at the bottom of `src/main.py`. Type `exit`, `quit`, or `bye` to leave.

## Dev container

A `.devcontainer/` is included (Python 3.11). On attach it installs `requirements.txt` and launches the Streamlit app on port `8501`. (You'll also need the spaCy/Presidio packages and `en_core_web_sm` model from the [Setup](#setup) step.)

## Notes

- The agent is prompted to keep answers under 30 words and to use each tool at most once per question (also hard-enforced for `retrieve_context`).
- Models used: `gpt-4.1-mini` (agent + eval generation), `gpt-4o` (image captioning), `text-embedding-3-small` (embeddings), `en_core_web_sm` (spaCy NER + Presidio).
- `.env`, `.venv/`, `data_files/`, `__pycache__/`, and `processed/` are gitignored.
