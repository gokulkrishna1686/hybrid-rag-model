"""Document ingestion pipeline (with per-file caching).

process_document turns an uploaded file into everything the agent needs to retrieve over:
parent + child chunks, an enriched parents map, a Chroma vector store, the extracted SQL
tables, and the chunk records. Each document is cached under processed/<md5-hash>/, so a
re-run on the same file reuses the cache instead of re-extracting / re-embedding /
re-captioning.
"""

import json
import os
import shutil
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document

from config import PROCESSED_DIR, file_hash
from chunking import split_parent_child
from enrich import enrich_text
from extract_text import load_text_docs
from extract_tables import extract_tables
from extract_images import (
    extract_and_caption_images,
    image_to_documents,
    descriptions_to_serializable,
    descriptions_from_serializable,
)


@dataclass
class ProcessedDoc:
    """Everything build_agent needs from a processed document."""
    doc_hash: str
    processed_dir: str
    eval_path: str
    chunks: list                 # child Documents (embedded + BM25-indexed)
    parents_by_id: dict          # parent_id -> enriched parent Document (what the LLM sees)
    chunk_records: list          # serializable child records (used by eval generation)
    vectorstore: Chroma
    table_metadata: list
    db: SQLDatabase


def _doc_paths(processed_dir):
    """All cache file/dir paths for one document's processed/<hash>/ folder."""
    return {
        "chunks": os.path.join(processed_dir, "chunks.json"),
        "parents": os.path.join(processed_dir, "parents.json"),
        "image_cache": os.path.join(processed_dir, "image_descriptions.json"),
        "eval": os.path.join(processed_dir, "eval_dataset.json"),
        "chroma": os.path.join(processed_dir, "chroma_db"),
        "db": os.path.join(processed_dir, "tables.db"),
        "table_meta": os.path.join(processed_dir, "table_metadata.json"),
        "images_dir": os.path.join(processed_dir, "images"),
    }


def _load_or_extract_tables(file_path, doc_hash, paths):
    """Load the cached SQLite tables + metadata, or extract them on a cache miss.
    Returns (table_metadata, db)."""
    if os.path.exists(paths["db"]) and os.path.exists(paths["table_meta"]):
        print(f"Tables already extracted (hash={doc_hash[:8]}). Loading cached .db + metadata.")
        with open(paths["table_meta"], "r", encoding="utf-8") as f:
            table_metadata = json.load(f)
    else:
        print(f"Extracting tables (hash={doc_hash[:8]})...")
        result = extract_tables(file_path, db_path=paths["db"])
        table_metadata = result["table_metadata"]
        with open(paths["table_meta"], "w", encoding="utf-8") as f:
            json.dump(table_metadata, f, indent=4, ensure_ascii=False)

    db = SQLDatabase.from_uri(
        f"sqlite:///{paths['db']}",
        sample_rows_in_table_info=0
    )
    return table_metadata, db


def _load_cached_index(doc_hash, paths, embeddings):
    """Rebuild chunks, parents_by_id, chunk_records, and the Chroma store from cache.
    Returns (chunks, parents_by_id, chunk_records, vectorstore)."""
    print(f"File already processed (hash={doc_hash[:8]}). Loading cached chunks + vectorstore.")
    with open(paths["chunks"], "r", encoding="utf-8") as f:
        chunk_records = json.load(f)
    with open(paths["parents"], "r", encoding="utf-8") as f:
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
        persist_directory=paths["chroma"],
        embedding_function=embeddings
    )
    return chunks, parents_by_id, chunk_records, vectorstore


def _process_and_index(file_path, doc_hash, paths, embeddings, api_key=None):
    """Cache miss: load text, split parent/child, caption images, enrich parents, persist
    the chunk/parent json, and (re)build the Chroma store. Returns
    (chunks, parents_by_id, chunk_records, vectorstore)."""
    print(f"Processing file (hash={doc_hash[:8]})...")
    docs = load_text_docs(file_path)

    chunks, parents_by_id = split_parent_child(docs)

    # captions are expensive (gpt-4o vision per image) and never change for
    # the same file, so cache them separately from the chunk/Chroma rebuild.
    if os.path.exists(paths["image_cache"]):
        print(f"Images already captioned (hash={doc_hash[:8]}). Loading cached captions.")
        with open(paths["image_cache"], "r", encoding="utf-8") as f:
            image_descriptions = descriptions_from_serializable(json.load(f))
    else:
        image_descriptions = extract_and_caption_images(
            file_path, output_folder=paths["images_dir"], api_key=api_key)
        with open(paths["image_cache"], "w", encoding="utf-8") as f:
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

    with open(paths["chunks"], "w", encoding="utf-8") as f:
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
    with open(paths["parents"], "w", encoding="utf-8") as f:
        json.dump(parent_records, f, indent=4, ensure_ascii=False)

    if os.path.exists(paths["chroma"]):
        shutil.rmtree(paths["chroma"])

    vectorstore = Chroma.from_documents(
        documents=chunks,           # children get embedded
        embedding=embeddings,
        persist_directory=paths["chroma"]
    )
    return chunks, parents_by_id, chunk_records, vectorstore


def process_document(file_path, embeddings, api_key=None):
    """Process a file (pdf/docx/pptx) into a ProcessedDoc, using the per-file cache when
    available. `embeddings` is the (caching) embeddings backend used for the Chroma store;
    `api_key` is used for image captioning (only when a fresh doc actually has images)."""
    doc_hash = file_hash(file_path)
    processed_dir = str(PROCESSED_DIR / doc_hash)
    paths = _doc_paths(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    table_metadata, db = _load_or_extract_tables(file_path, doc_hash, paths)

    index_cached = (
        os.path.exists(paths["chunks"])
        and os.path.exists(paths["chroma"])
        and os.path.exists(paths["parents"])
    )
    if index_cached:
        chunks, parents_by_id, chunk_records, vectorstore = _load_cached_index(
            doc_hash, paths, embeddings
        )
    else:
        chunks, parents_by_id, chunk_records, vectorstore = _process_and_index(
            file_path, doc_hash, paths, embeddings, api_key
        )

    return ProcessedDoc(
        doc_hash=doc_hash,
        processed_dir=processed_dir,
        eval_path=paths["eval"],
        chunks=chunks,
        parents_by_id=parents_by_id,
        chunk_records=chunk_records,
        vectorstore=vectorstore,
        table_metadata=table_metadata,
        db=db,
    )
