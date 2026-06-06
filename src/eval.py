"""Run questions through the hybrid-RAG agent and return one dict per query.

Reuses main.build_agent so the agent, retriever, and RBAC are identical to the
app. For each query it returns a JSON-serializable dict for evaluation:

    {
        "question": "...",            # the query asked
        "response": "...",            # the agent's answer
        "chunks_retrieved": [...],    # child chunk_ids matched (vs expected_contexts)
        "tables_queried": [...],      # table names the SQL agent ran (vs expected_contexts)
        "sources": [...],             # hash of the source file the answer drew from
    }

run_eval() writes exactly this dict per question to eval_results.json. The gold
fields (ground_truth, answer_type, keywords, expected_contexts) are NOT copied in
— they live in eval_dataset.json and are joined back by the question string.

A table-only question retrieves no context, so the retrieval fields come back
empty.
"""

import json
import os
import random
import re

from main import build_agent, file_hash, PROCESSED_DIR, DATA_DIR

# tables are named "page_X_table_Y", so we can recover which table the SQL agent
# queried by scanning the SQL it ran for that pattern.
_TABLE_RE = re.compile(r"page_\d+_table_\d+", re.IGNORECASE)


def _tables_from_messages(messages):
    """Table names the SQL agent actually queried, parsed from its tool calls.

    Only query_pdf_tables tool-call args are inspected (the executed SQL) — not
    get_table_metadata output, which lists every table and would be a false match.
    """
    tables = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") != "query_pdf_tables":
                continue
            sql = (tc.get("args") or {}).get("query", "")
            for name in _TABLE_RE.findall(sql):
                name = name.lower()
                if name not in tables:
                    tables.append(name)
    return tables


def answer_query(agent, retrieval_log, reset_turn, query):
    """Run one query through the agent and return an evaluation dict."""
    before = len(retrieval_log)

    reset_turn()  # reset the per-turn retrieve_context guard
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    answer = result["structured_response"].message

    # at most one retrieval entry is appended per turn (the guard caches repeats);
    # a turn may retrieve nothing (e.g. table-only questions).
    new_entries = retrieval_log[before:]
    entry = new_entries[-1] if new_entries else None

    return {
        "question": query,
        "response": answer,
        "chunks_retrieved": entry["child_matches"] if entry else [],
        "tables_queried": _tables_from_messages(result.get("messages", [])),
        "sources": entry["sources"] if entry else [],
    }


def run_cli(file_path, role="employee"):
    """Interactive REPL: type a query in the terminal, get the eval dict printed
    and saved to processed/<hash>/eval_results.json.

    Same caching as the app — if the file's already processed (chunks + Chroma
    + tables exist for its hash), build_agent loads them instead of rebuilding.
    """
    agent, retrieval_log, reset_turn = build_agent(
        file_path, generate_eval=False, role=role
    )

    results_path = os.path.join(
        PROCESSED_DIR, file_hash(file_path), "eval_results.json"
    )
    results = []

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Exiting...")
            break

        record = answer_query(agent, retrieval_log, reset_turn, query)
        print(json.dumps(record, indent=2, ensure_ascii=False))

        # rewrite the full list each turn so the file stays valid JSON
        results.append(record)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"(saved -> {results_path})")


def run_eval(file_path, role="employee", k=None):
    """Answer questions from eval_dataset.json and save the answers to
    eval_results.json (one bare answer dict per question).

    k: how many questions to answer. Default (None) answers ALL of them; otherwise
    a random sample of k questions is answered (capped at the dataset size).

    Skips entirely if eval_results.json already exists for this file's hash:
    the existing results are loaded and returned — no agent build, no API calls.
    """
    processed_dir = os.path.join(PROCESSED_DIR, file_hash(file_path))
    eval_path = os.path.join(processed_dir, "eval_dataset.json")
    results_path = os.path.join(processed_dir, "eval_results.json")

    # already computed -> skip the whole run (no agent build, no API calls)
    if os.path.exists(results_path):
        print(f"eval_results.json already exists -> skipping ({results_path})")
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # build the agent (generate_eval=True also creates eval_dataset.json if it is
    # missing, so we always have questions to read below)
    agent, retrieval_log, reset_turn = build_agent(
        file_path, generate_eval=True, role=role
    )

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    # k=None -> all questions; otherwise a random sample of k (capped at the total)
    if k is not None and k < len(eval_items):
        eval_items = random.sample(eval_items, k)
        print(f"Answering a random sample of {k} question(s)")

    results = []
    for i, item in enumerate(eval_items, 1):
        print(f"\n[{i}/{len(eval_items)}] {item['question']}")
        record = answer_query(agent, retrieval_log, reset_turn, item["question"])
        results.append(record)

        # rewrite the full list each turn so progress is durable even if a later
        # question errors out (and the file always stays valid JSON).
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nEval results saved to {results_path}")
    return results


if __name__ == "__main__":
    file_name = str(DATA_DIR / "Employee Performance.docx")
    # batch: answer questions from eval_dataset.json and save them to
    # eval_results.json. Skipped automatically if eval_results.json already exists.
    # Pass k=N to answer only a random sample of N questions (default: all).
    # Use run_cli(file_name) instead to ask questions interactively.
    run_eval(file_name, role="manager", k=3)