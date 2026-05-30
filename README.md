# Hybrid RAG Model

A retrieval-augmented question-answering system for company report PDFs. It combines **three retrieval paths** behind a single LangChain agent so that each kind of question is answered by the tool best suited to it:

1. **Keyword retrieval (BM25)** — exact-term and rare-keyword matches.
2. **Dense semantic retrieval (Chroma + OpenAI embeddings)** — paraphrased / conceptual questions.
3. **SQL over extracted tables** — calculations, aggregations, filters, and exact lookups on structured data pulled out of the PDF.

Text, image captions, and tables are all extracted from the uploaded PDF, and the agent decides which path to use per question.

## How it works

When a PDF is processed (`build_agent` in `main.py`), the pipeline:

1. **Splits text** into chunks with `RecursiveCharacterTextSplitter` (400 chars, 50 overlap).
2. **Extracts & captions images** (`extract_images.py`) using PyMuPDF (`fitz`) to pull raster images and GPT-4o vision to produce a structured caption (type, summary, per-entity colors/shapes/text, keywords). Each image becomes one *summary* document plus one document per detected *entity*.
3. **Extracts tables** (`extract_tables.py`) with `pdfplumber`, cleans column names and cells, auto-detects numeric columns, and writes one SQLite table per detected table (`page_<n>_table_<i>`).
4. **Builds retrievers**: a `BM25Retriever` and a Chroma vector store (`text-embedding-3-small`) are combined in an `EnsembleRetriever` (BM25 weight `0.4`, semantic weight `0.6`, fused with Reciprocal Rank Fusion).
5. **Creates the agent** (`create_agent`, model `gpt-4.1-mini`) exposing three tools:
   - `retrieve_context(query)` — hybrid BM25 + semantic retrieval for prose, summaries, and image content.
   - `get_table_metadata()` — returns table schemas and sample rows so the agent can pick the right table.
   - `query_pdf_tables(query)` — runs a SQLite query against the extracted tables.

### Caching

Processing is cached per PDF by MD5 hash under `processed/<hash>/` (chunks, Chroma DB, SQLite tables, table metadata, optional eval dataset). Re-running on the same file reuses the cached artifacts instead of re-extracting.

### Evaluation dataset (optional)

When `generate_eval=True` (used by the CLI), an LLM generates a mixed evaluation set — keyword, conceptual, paraphrased, and table questions — saved to `processed/<hash>/eval_dataset.json` to stress all three retrieval paths.

## Project structure

```
main.py             # Pipeline + agent (build_agent) and CLI entry point
streamlit_app.py    # Streamlit chat UI with PDF upload
extract_images.py   # Image extraction (PyMuPDF) + GPT-4o vision captioning
extract_tables.py   # Table extraction (pdfplumber) into SQLite
notebooks/          # Exploratory notebooks (BM25/Chroma, table extraction)
requirements.txt    # Pinned dependencies
processed/          # Per-PDF cache (gitignored)
data_files/         # Local PDFs (gitignored)
```

## Setup

Requires Python 3.11 and an OpenAI API key.

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Usage

### Streamlit app

```powershell
streamlit run streamlit_app.py
```

Upload a PDF in the sidebar, wait for processing, then ask questions in the chat.

### Command line

`main.py` runs an interactive CLI against a hard-coded PDF path:

```powershell
python main.py
```

By default it loads `data_files/Employee Performance.pdf` — edit the `file_name` value at the bottom of `main.py` to point at your own PDF. Type `exit`, `quit`, or `bye` to leave.

## Dev container

A `.devcontainer/` is included (Python 3.11). On attach it installs `requirements.txt` and launches the Streamlit app on port `8501`.

## Notes

- The agent is prompted to keep answers under 30 words and to use each tool at most once per question.
- Models used: `gpt-4.1-mini` (agent + eval generation), `gpt-4o` (image captioning), `text-embedding-3-small` (embeddings).
- `.env`, `.venv/`, `data_files/`, `__pycache__/`, and `processed/` are gitignored.
