"""Parent-child text splitting: large parent chunks carry enough context to answer with;
small child chunks are what we embed and BM25-index for precise matching. Each child
stores the parent_id it came from so we can swap it back at retrieval time."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)


def split_parent_child(
    docs,
    parent_chunk_size=PARENT_CHUNK_SIZE,
    parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
    child_chunk_size=CHILD_CHUNK_SIZE,
    child_chunk_overlap=CHILD_CHUNK_OVERLAP,
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
