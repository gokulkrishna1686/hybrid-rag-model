"""RBAC enforcement on retrieved parent chunks: keep what's within the caller's clearance,
and redact (don't drop) what's above it, wrapping it in a [RESTRICTED] marker."""

from langchain_core.documents import Document

from enrich import redact, SENSITIVITY_ORDER


def apply_clearance(parent_docs, clearance, role):
    """Split retrieved parents into what the role may see vs. what it may not. Docs at or
    below clearance pass through unchanged; docs above clearance are PII-masked and wrapped
    in a [RESTRICTED] marker (so the agent treats them as an access block, not a retrieval
    miss). Returns (visible_docs, redacted_ids)."""
    visible_docs = []
    redacted_ids = []
    for p in parent_docs:
        level = SENSITIVITY_ORDER.get(p.metadata.get("sensitivity", "public"), 0)
        if level <= clearance:
            visible_docs.append(p)
        else:
            # too sensitive for this role -> mask the PII, keep the prose, and ANNOUNCE
            # the redaction so the agent treats it as an access block (and stops) rather
            # than a retrieval miss (and retries).
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
    return visible_docs, redacted_ids


def serialize_docs(visible_docs):
    """Join the visible docs into the single text block handed to the LLM."""
    return "\n\n".join(
        (
            f"Source: {doc.metadata}\n"
            f"Content: {doc.page_content}"
        )
        for doc in visible_docs
    )
