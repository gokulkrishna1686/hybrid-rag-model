"""CLI entry point for the hybrid-RAG agent.

The implementation now lives in focused modules — config, chunking, embeddings, pipeline,
retrievers, rbac, tools, eval_dataset, agent — and this file is just the interactive chat
loop. The public names below are re-exported so existing `from main import ...` callers
(notebooks, scratch, older scripts) keep working.
"""

from agent import build_agent, build_retrievers
from config import file_hash, PROCESSED_DIR, DATA_DIR

__all__ = [
    "build_agent",
    "build_retrievers",
    "file_hash",
    "PROCESSED_DIR",
    "DATA_DIR",
    "run_cli",
]


def run_cli(file_path, role="employee"):
    agent, _, reset_turn = build_agent(file_path, generate_eval=True, role=role)

    while True:
        user_query = input("\nYou: ")

        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Exiting...")
            break

        reset_turn()
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
    file_name = str(DATA_DIR / "Employee Performance.docx")
    # role controls what the retriever is allowed to surface:
    #   guest -> public, employee -> internal, manager -> confidential
    run_cli(file_name, role="manager")
