"""An embeddings wrapper that memoizes embed_query (optionally persisting to disk)."""

import json
import os

from langchain_core.embeddings import Embeddings


class CachingEmbeddings(Embeddings):
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
