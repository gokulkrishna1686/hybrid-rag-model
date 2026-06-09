import os
import tempfile

import streamlit as st

from agent import build_agent
from config import DATA_DIR
from extract_text import load_text_docs
from enrich import ROLE_CLEARANCE, SENSITIVITY_ORDER, STRONG_PII, MILD_PII

DEMO_DOC = DATA_DIR / "Tier Demo.docx"  # built-in demo: one section per sensitivity tier


st.set_page_config(page_title="Hybrid RAG Chat", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# Access-control (RBAC) presentation.
# All of this is DERIVED from enrich.py (ROLE_CLEARANCE, SENSITIVITY_ORDER,
# STRONG_PII, MILD_PII), so the UI explanation can never drift from the policy
# the pipeline actually enforces.
# ---------------------------------------------------------------------------
ROLES = list(ROLE_CLEARANCE.keys())  # guest, employee, manager

# Human-readable names for the Presidio PII types the pipeline classifies on.
PII_LABELS = {
    "PERSON": "Names of people",
    "LOCATION": "Locations",
    "NRP": "Nationality / religion / politics",
    "EMAIL_ADDRESS": "Email addresses",
    "PHONE_NUMBER": "Phone numbers",
    "CREDIT_CARD": "Credit-card numbers",
    "US_SSN": "Social Security numbers",
    "IBAN_CODE": "Bank IBANs",
    "US_BANK_NUMBER": "Bank-account numbers",
    "US_DRIVER_LICENSE": "Driver's licenses",
    "US_PASSPORT": "Passport numbers",
    "US_ITIN": "Taxpayer IDs (ITIN)",
    "MEDICAL_LICENSE": "Medical licenses",
    "CRYPTO": "Crypto wallet addresses",
    "IP_ADDRESS": "IP addresses",
}

# Which PII puts a chunk into each tier (mirrors enrich.classify_sensitivity:
# strong PII -> confidential, mild PII -> internal, nothing -> public).
TIER_PII = {
    "public": set(),
    "internal": MILD_PII,
    "confidential": STRONG_PII,
}
TIER_EMOJI = {"public": "🟢", "internal": "🟡", "confidential": "🔴"}
TIERS = sorted(SENSITIVITY_ORDER, key=SENSITIVITY_ORDER.get)  # least -> most sensitive


def friendly(pii_types):
    """Comma-joined human labels for a set of Presidio PII types."""
    return ", ".join(sorted(PII_LABELS.get(t, t) for t in pii_types))


def roles_for_tier(tier):
    """Which roles are cleared to see content tagged at this tier."""
    need = SENSITIVITY_ORDER[tier]
    return [r for r in ROLES if SENSITIVITY_ORDER[ROLE_CLEARANCE[r]] >= need]


def visible_and_masked(role):
    """(visible_pii, masked_pii) for a role across every tier."""
    clearance = SENSITIVITY_ORDER[ROLE_CLEARANCE.get(role, "public")]
    visible, masked = set(), set()
    for tier, pii in TIER_PII.items():
        (visible if SENSITIVITY_ORDER[tier] <= clearance else masked).update(pii)
    return visible, masked


def render_rbac_help():
    """The shared 'how access control works' explainer (sidebar expander)."""
    st.markdown(
        "Every chunk of the document is scanned for **PII** and tagged with a "
        "**sensitivity tier**. Your **role** sets a clearance level — you see your "
        "tier and everything below it. Content above your level isn't deleted: its "
        "PII is **masked** (e.g. `<PERSON>`, `<EMAIL_ADDRESS>`) and marked "
        "`[RESTRICTED]`, so the assistant tells you you're not authorized rather than "
        "pretending the content doesn't exist."
    )
    rows = [
        f"| {TIER_EMOJI[t]} **{t}** "
        f"| {friendly(TIER_PII[t]) or 'none — general text, no PII'} "
        f"| {', '.join(roles_for_tier(t))} |"
        for t in TIERS
    ]
    st.markdown(
        "| Tier | Triggered by (PII found) | Roles allowed |\n"
        "|---|---|---|\n" + "\n".join(rows)
    )


@st.cache_data(show_spinner=False)
def load_demo_text(path):
    """Extracted text of the demo document (cached so it's read only once)."""
    return "\n\n".join(d.page_content for d in load_text_docs(path))


# Starter questions for the demo, spanning the three tiers so switching roles
# visibly changes what the assistant will answer.
DEMO_QUESTIONS = [
    "What are the stages of the performance review cycle?",            # public — any role
    "Who led the product team review, and at which office?",           # internal — employee+
    "What is Priya Sharma's email and social security number?",        # confidential — manager
    "What is the corporate card number assigned to the travel desk?",  # confidential — manager
]


def render_demo_guide(show_questions):
    """Show what's inside the demo doc + clickable starter questions, so users know what
    to ask. Clicking a question stashes it in session for the chat loop to pick up."""
    st.markdown(
        "📄 **Demo — Performance Review.** Three sections of rising sensitivity: a "
        "🟢 **public** framework, 🟡 **internal** reviewer notes (names & offices), and a "
        "🔴 **confidential** contact section (emails, phones, SSNs, a card number). "
        "Switch roles in the sidebar and re-ask to watch access control kick in."
    )
    with st.expander("📄 Read the demo document"):
        st.markdown(load_demo_text(str(DEMO_DOC)).replace("\n", "\n\n"))
    if show_questions:
        st.caption("Try one — then change your role and ask again:")
        for q in DEMO_QUESTIONS:
            if st.button(q, key=f"demoq::{q}", width="stretch"):
                st.session_state.pending_query = q


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.title("Hybrid RAG Chat")
st.caption(
    "Ask questions about an uploaded report. Answers respect your access level — "
    "PII above your clearance is masked before the assistant ever sees it."
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None


def rebuild_agent():
    """(Re)build the agent for the current document + role + key, and reset the chat.
    Returns True on success, False if it couldn't build (missing inputs or an API error)."""
    path = st.session_state.get("doc_path")
    role = st.session_state.get("role", "employee")
    key = st.session_state.get("openai_key")
    if not path or not key:
        return False
    try:
        with st.spinner(f"Processing as role '{role}'... this may take a minute."):
            agent, retrieval_log, reset_turn = build_agent(path, role=role, api_key=key)
    except Exception as e:
        st.error(f"Could not build the agent — {e}")
        return False
    st.session_state.agent = agent
    st.session_state.retrieval_log = retrieval_log
    st.session_state.reset_turn = reset_turn
    st.session_state.messages = []
    st.session_state.last_retrieval = None
    return True


with st.sidebar:
    st.header("OpenAI API key")
    api_key = st.text_input(
        "Your OpenAI API key",
        type="password",
        value=st.session_state.get("openai_key", ""),
        placeholder="sk-...",
        help="Used only for this session's requests (billed to your account). "
             "Kept in memory, never written to disk.",
    )
    st.session_state["openai_key"] = api_key
    if not api_key:
        st.caption("🔑 Required — bring your own key; usage is billed to you.")

    st.header("1 · Document")
    source = st.radio(
        "Pick a document",
        ["Demo (Tier Demo.docx)", "Upload your own"],
        captions=[
            "Built-in doc with one section per sensitivity tier — great for trying roles.",
            "Your own PDF, DOCX, or PPTX.",
        ],
    )

    doc_path = None
    doc_name = None
    if source.startswith("Demo"):
        if DEMO_DOC.exists():
            doc_path, doc_name = str(DEMO_DOC), DEMO_DOC.name
        else:
            st.warning(f"Demo file not found: {DEMO_DOC}")
    else:
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "pptx"])
        if uploaded_file is not None:
            # save a newly-uploaded file to a temp path (once)
            if st.session_state.get("uploaded_name") != uploaded_file.name:
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_path = tmp_path
                st.session_state.uploaded_name = uploaded_file.name
            doc_path, doc_name = st.session_state.uploaded_path, uploaded_file.name

    st.header("2 · Access level (RBAC)")
    role = st.selectbox(
        "Role",
        ROLES,
        index=ROLES.index(st.session_state.get("role", "employee")),
        help="Your role sets how much of the document you're cleared to see.",
    )

    tier = ROLE_CLEARANCE[role]
    st.caption(
        f"Clearance tier: {TIER_EMOJI[tier]} **{tier}** (level {SENSITIVITY_ORDER[tier]})"
    )

    visible, masked = visible_and_masked(role)
    st.markdown(
        f"✅ **You can see:** {friendly(visible) if visible else 'general report text only (no PII)'}"
    )
    st.markdown(
        f"🔒 **Masked for you:** {friendly(masked) if masked else 'nothing — full access'}"
    )

    with st.expander("How access control works"):
        render_rbac_help()

    # build once we have BOTH a document and a key
    if doc_path is None:
        if source.startswith("Upload"):
            st.info("⬆️ Upload a file to continue.")
    elif not api_key:
        st.info("🔑 Enter your OpenAI API key above to start.")
    else:
        # (re)build whenever the document, role, or key changes
        current = (doc_name, role, api_key)
        if st.session_state.get("built") != current:
            st.session_state.doc_path = doc_path
            st.session_state.role = role
            if rebuild_agent():
                st.session_state.built = current
        if st.session_state.get("built") == current:
            st.success(f"Ready: {doc_name}  ·  role: {role}")


# chat on the left, retrieval details on the right
chat_col, info_col = st.columns([2, 1])

with chat_col:
    if doc_name == DEMO_DOC.name:
        render_demo_guide(show_questions=("agent" in st.session_state and not st.session_state.messages))
    elif "agent" not in st.session_state:
        st.info("👈 Add your OpenAI API key and pick a document (the demo works out of the box) to begin.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed = st.chat_input("Ask a question about the document...")
    user_query = typed or st.session_state.pop("pending_query", None)

    if user_query:
        if "agent" not in st.session_state:
            st.warning("Add your OpenAI API key and pick a document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            log = st.session_state.retrieval_log
            before = len(log)

            st.session_state.reset_turn()   # reset the per-turn retrieve guard

            answer = None
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.agent.invoke(
                            {"messages": [{"role": "user", "content": user_query}]}
                        )
                        answer = result["structured_response"].message
                    except Exception as e:
                        st.error(f"OpenAI request failed — {e}. Check your API key / billing.")
                if answer is not None:
                    st.markdown(answer)

            if answer is not None:
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
            active_role = st.session_state.get("role", role)
            n = len(entry["redacted"])
            st.warning(
                f"🔒 {n} chunk(s) were above your **{active_role}** clearance — their PII "
                f"was masked (e.g. `<PERSON>`, `<EMAIL_ADDRESS>`) and marked `[RESTRICTED]` "
                f"before the assistant saw them."
            )
            st.caption("Restricted parents: " + ", ".join(entry["redacted"]))
