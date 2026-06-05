"""Load (or build) a document's eval dataset and eval results as two dicts.

Usage:
    from test_eval import load_eval_data
    eval_dataset, eval_results = load_eval_data("data_files/Employee Performance.pdf")

Both dicts are keyed by the question string (the join key — run_eval copies each
dataset question verbatim into its result, so the two match exactly, no index needed)
and are restricted to the questions present in BOTH files, so they line up 1:1
(e.g. if eval_results only has 3 answered questions, eval_dataset is trimmed to the
same 3):
    eval_dataset[question] -> gold item   {ground_truth, answer_type, keywords, expected_contexts}
    eval_results[question] -> eval record {response, chunks_retrieved, tables_queried, sources, ...}

No redundant work / API calls: if processed/<hash>/eval_dataset.json and
eval_results.json already exist, they are just loaded. Otherwise only the missing
one is generated — the dataset via main.build_agent(generate_eval=True), the results
via eval.run_eval (which itself generates the dataset too if it is missing).
"""

import json
import os

from main import build_agent, file_hash
from eval import run_eval


def load_eval_data(file_path, role="employee"):
    """Return (eval_dataset, eval_results) for `file_path` as dicts keyed by question.

    Each file is checked independently and ONLY the missing one is generated:
      - eval_dataset.json missing -> build_agent(generate_eval=True) makes just the dataset
      - eval_results.json missing -> run_eval makes just the results (reusing the dataset)
    If both exist, nothing is generated (no API calls). If both are missing, run_eval
    creates the dataset and then the results in a single agent build.
    """
    processed_dir = os.path.join("processed", file_hash(file_path))
    dataset_path = os.path.join(processed_dir, "eval_dataset.json")
    results_path = os.path.join(processed_dir, "eval_results.json")

    have_dataset = os.path.exists(dataset_path)
    have_results = os.path.exists(results_path)

    if not have_dataset and not have_results:
        # neither exists -> run_eval generates the dataset, then the results (one build).
        print("eval_dataset.json + eval_results.json missing -> generating both")
        run_eval(file_path, role=role)
    elif not have_dataset:
        # only the dataset is missing -> generate JUST the dataset.
        print("eval_dataset.json missing -> generating dataset only")
        build_agent(file_path, generate_eval=True, role=role)
    elif not have_results:
        # only the results are missing -> generate JUST the results (dataset is reused).
        print("eval_results.json missing -> generating results only")
        run_eval(file_path, role=role)
    else:
        # both already present -> generate nothing, no API calls.
        print("eval_dataset.json + eval_results.json present -> loading (no generation)")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_items = json.load(f)
    with open(results_path, "r", encoding="utf-8") as f:
        result_records = json.load(f)

    full_dataset = {item["question"]: item for item in dataset_items}
    full_results = {rec["question"]: rec for rec in result_records}

    # keep only questions present in BOTH files (eval_results may be a subset of the
    # dataset, e.g. when run_eval answered only k questions) so the two dicts align 1:1
    common = full_dataset.keys() & full_results.keys()
    eval_dataset = {q: full_dataset[q] for q in common}
    eval_results = {q: full_results[q] for q in common}

    return eval_dataset, eval_results


if __name__ == "__main__":
    file_name = "data_files/Employee Performance.docx"
    eval_dataset, eval_results = load_eval_data(file_name)
    print(f"eval_dataset: {len(eval_dataset)} questions")
    print(f"eval_results: {len(eval_results)} questions")
    