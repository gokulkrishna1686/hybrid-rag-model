import os
import tempfile

import streamlit as st

from agent import build_agent
from enrich import ROLE_CLEARANCE


st.set_page_config(page_title="Hybrid RAG Chat", page_icon="🤖", layout="wide")
st.title("Hybrid RAG Chat")

ROLES = list(ROLE_CLEARANCE.keys())  # guest, employee, manager, hr, admin

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None


def rebuild_agent():
    """(Re)build the agent for the current document + role, and reset the chat."""
    path = st.session_state.get("doc_path")
    role = st.session_state.get("role", "employee")
    if not path:
        return
    with st.spinner(f"Processing as role '{role}'... this may take a minute."):
        agent, retrieval_log, reset_turn = build_agent(path, role=role)
    st.session_state.agent = agent
    st.session_state.retrieval_log = retrieval_log
    st.session_state.reset_turn = reset_turn
    st.session_state.messages = []
    st.session_state.last_retrieval = None


with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "pptx"])

    st.header("Access level (RBAC)")
    role = st.selectbox(
        "Role",
        ROLES,
        index=ROLES.index(st.session_state.get("role", "employee")),
    )
    st.caption(f"Clearance tier: **{ROLE_CLEARANCE[role]}**")

    if uploaded_file is not None:
        # new upload -> save to a temp file and build
        if st.session_state.get("doc_name") != uploaded_file.name:
            tmp_dir = tempfile.mkdtemp()
            doc_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(doc_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.doc_path = doc_path
            st.session_state.doc_name = uploaded_file.name
            st.session_state.role = role
            rebuild_agent()
        # same document but the role changed -> rebuild with new clearance
        elif st.session_state.get("role") != role:
            st.session_state.role = role
            rebuild_agent()

        st.success(f"Ready: {uploaded_file.name}  ·  role: {role}")


# chat on the left, retrieval details on the right
chat_col, info_col = st.columns([2, 1])

with chat_col:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about the document...")

    if user_query:
        if "agent" not in st.session_state:
            st.warning("Please upload a document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            log = st.session_state.retrieval_log
            before = len(log)

            st.session_state.reset_turn()   # reset the per-turn retrieve guard

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.agent.invoke(
                        {"messages": [{"role": "user", "content": user_query}]}
                    )
                    answer = result["structured_response"].message
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

            # capture the retrieval this turn produced (a turn may not retrieve)
            new_entries = log[before:]
            st.session_state.last_retrieval = new_entries[-1] if new_entries else None

with info_col:
    st.subheader("Retrieval details")
    entry = st.session_state.last_retrieval

    if entry is None:
        st.caption(
            "Scores appear here when the assistant retrieves document context. "
            "Table-only questions won't trigger retrieval."
        )
    else:
        st.markdown(f"**Query sent to retriever:** {entry['query']}")
        if entry.get("must_mention"):
            st.markdown(f"**Entity filter (`must_mention`):** `{entry['must_mention']}`")

        st.markdown("**BM25 — keyword scores**")
        st.dataframe(entry["bm25"], hide_index=True, width="stretch")

        st.markdown("**Semantic — vector distance** (lower = closer)")
        st.dataframe(entry["semantic"], hide_index=True, width="stretch")

        st.markdown("**RRF — fused re-ranking** (final order)")
        st.dataframe(entry["rrf"], hide_index=True, width="stretch")

        parents = ", ".join(entry["parents"]) if entry["parents"] else "—"
        st.markdown(f"**Parents returned to LLM:** {parents}")

        if entry.get("redacted"):
            st.warning(
                "Redacted (above your clearance): " + ", ".join(entry["redacted"])
            )
