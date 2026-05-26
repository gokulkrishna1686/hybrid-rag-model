from langchain_community.document_loaders import PyPDFLoader
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
import shutil
import hashlib
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
import json
import random

from extract_images import extract_and_caption_images, image_to_documents
from extract_tables import extract_pdf_tables

load_dotenv()
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

3. retrieve_context(query: str)
- Use ONLY for semantic document retrieval, summaries, explanations, charts, images, and general report context.
- DO NOT use this for exact table calculations or SQL-style queries.
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


def _generate_eval_dataset(eval_path, chunk_records, table_metadata, db):
    print("Generating eval dataset (LLM)...")

    class ExpectedContext(BaseModel):
        chunk_id: str | None = None
        table_name: str | None = None

    class EvalItem(BaseModel):
        question: str
        ground_truth: str
        expected_contexts: list[ExpectedContext]

    class EvalDataset(BaseModel):
        items: list[EvalItem]

    eval_llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3,
        api_key=api_key,
    ).with_structured_output(EvalDataset)

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

    eval_prompt = f"""You are generating an evaluation dataset for a hybrid RAG system
that combines BM25 (keyword) retrieval, dense semantic retrieval, and a SQL agent
that queries structured tables extracted from the PDF. The eval set must stress all
three paths.

Generate {num_questions} diverse evaluation questions total:
- {num_chunk_questions} questions answerable from the CHUNKS (free-text prose + image captions)
- {num_table_questions} questions answerable from the TABLES (SQL aggregations, lookups, filters)

For each item provide:
- question: a clear, specific question answerable strictly from the provided material
- ground_truth: the concise correct answer (a value, name, phrase, or 1-2 sentence explanation)
- expected_contexts: a list of context references. Each entry is an object with EITHER:
    * {{"chunk_id": "chunk_N"}}  — when the answer lives in a chunk
    * {{"table_name": "page_X_table_Y"}}  — when the answer requires querying a table
  A single question may cite multiple contexts, and may mix chunks AND tables when
  the answer is grounded in both.

CHUNK question mix (aim for roughly this distribution within the chunk questions):
- ~30% factual / numerical lookups that reuse rare terms verbatim — good for BM25.
- ~40% explanatory / conceptual questions ("why", "how", "what does X mean",
  "describe the process for Y"). Answers should be 1-2 sentence explanations.
- ~30% paraphrased / semantic questions where the wording deliberately AVOIDS
  the exact keywords used in the chunk (synonyms, rephrasing, indirect framing).
  These stress the dense retriever — BM25 should struggle on these.

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

    def serialize_context(ctx):
        out = {}
        if ctx.chunk_id:
            out["chunk_id"] = ctx.chunk_id
        if ctx.table_name:
            out["table_name"] = ctx.table_name
        return out

    eval_records = [
        {
            "question": item.question,
            "ground_truth": item.ground_truth,
            "expected_contexts": [
                serialize_context(c) for c in item.expected_contexts
                if c.chunk_id or c.table_name
            ]
        }
        for item in eval_data.items
    ]

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=4, ensure_ascii=False)
    print(f"Eval dataset saved to {eval_path}")


def build_agent(pdf_path, *, generate_eval=False):
    """Process a PDF (with caching) and return a ready-to-use agent."""
    pdf_hash = file_hash(pdf_path)
    processed_dir = os.path.join("processed", pdf_hash)
    chunks_path = os.path.join(processed_dir, "chunks.json")
    eval_path = os.path.join(processed_dir, "eval_dataset.json")
    chroma_path = os.path.join(processed_dir, "chroma_db")
    db_path = os.path.join(processed_dir, "tables.db")
    table_meta_path = os.path.join(processed_dir, "table_metadata.json")
    os.makedirs(processed_dir, exist_ok=True)

    if os.path.exists(db_path) and os.path.exists(table_meta_path):
        print(f"Tables already extracted (hash={pdf_hash[:8]}). Loading cached .db + metadata.")
        with open(table_meta_path, "r", encoding="utf-8") as f:
            table_metadata = json.load(f)
    else:
        print(f"Extracting tables (hash={pdf_hash[:8]})...")
        result = extract_pdf_tables(pdf_path, db_path=db_path)
        table_metadata = result["table_metadata"]
        with open(table_meta_path, "w", encoding="utf-8") as f:
            json.dump(table_metadata, f, indent=4, ensure_ascii=False)

    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        sample_rows_in_table_info=0
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    if os.path.exists(chunks_path) and os.path.exists(chroma_path):
        print(f"PDF already processed (hash={pdf_hash[:8]}). Loading cached chunks + vectorstore.")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_records = json.load(f)

        chunks = [
            Document(
                page_content=rec["text"],
                metadata={
                    "source": rec["document"],
                    "page": rec["page"],
                    "start_index": rec["start_index"],
                    "chunk_id": rec["chunk_id"],
                }
            )
            for rec in chunk_records
        ]

        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
    else:
        print(f"Processing PDF (hash={pdf_hash[:8]})...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            add_start_index=True
        )

        chunks = text_splitter.split_documents(docs)

        images_dir = os.path.join(processed_dir, "images")
        image_descriptions = extract_and_caption_images(pdf_path, output_folder=images_dir)

        image_chunks = []
        for item in image_descriptions:
            image_chunks.extend(image_to_documents(item))

        chunks.extend(image_chunks)

        chunk_records = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            chunk.metadata["chunk_id"] = chunk_id
            chunk_records.append({
                "chunk_id": chunk_id,
                "document": chunk.metadata.get('source'),
                "page": chunk.metadata.get('page'),
                "start_index": chunk.metadata.get('start_index'),
                "type": chunk.metadata.get('type'),
                "image_id": chunk.metadata.get('image_id'),
                "text": chunk.page_content
            })

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_records, f, indent=4, ensure_ascii=False)

        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)

        vectorstore = Chroma.from_documents(
            documents=chunks,
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

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            semantic_retriever
        ],
        weights=[0.4, 0.6]
    )

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
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""

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

        retrieved_docs = hybrid_retriever.invoke(query)

        type(retrieved_docs)

        retrieved_ids = [
            doc.metadata.get("chunk_id", "?") for doc in retrieved_docs
        ]
        print("CONTEXT RETRIEVED:", retrieved_ids)

        serialized = "\n\n".join(
            (
                f"Source: {doc.metadata}\n"
                f"Content: {doc.page_content}"
            )
            for doc in retrieved_docs
        )

        return serialized, retrieved_docs

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

    return create_agent(
        model=model,
        tools=tools,
        response_format=ToolStrategy(ResponseFormat),
        system_prompt=prompt
    )


def run_cli(pdf_path):
    agent = build_agent(pdf_path, generate_eval=True)

    while True:
        user_query = input("\nYou: ")

        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Exiting...")
            break

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
    file_name = "data_files/Employee Performance.pdf"
    run_cli(file_name)
