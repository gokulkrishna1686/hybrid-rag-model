import os
import tempfile

import streamlit as st

from main import build_agent


st.set_page_config(page_title="Hybrid RAG Chat", page_icon="🤖")
st.title("Hybrid RAG Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.session_state.get("pdf_name") != uploaded_file.name:
            with st.spinner("Processing PDF... this may take a minute."):
                tmp_dir = tempfile.mkdtemp()
                pdf_path = os.path.join(tmp_dir, uploaded_file.name)

                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.agent = build_agent(pdf_path)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []

            st.success(f"Ready: {uploaded_file.name}")
        else:
            st.success(f"Ready: {uploaded_file.name}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about the PDF...")

if user_query:
    if "agent" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.agent.invoke(
                    {
                        "messages": [
                            {"role": "user", "content": user_query}
                        ]
                    }
                )
                answer = result["structured_response"].message

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
