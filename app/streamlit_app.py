import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app.ingestion import load_pdf
from app.vectorstore import VectorStore
from app.rag import ask

st.set_page_config(page_title="Multi-Document RAG", page_icon="📄", layout="wide")
st.title("📄 Multi-Document RAG Chat")


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


def show_sources(sources):
    with st.expander("Sources"):
        for c in sources:
            st.markdown(f"**{c['source']}, p.{c['page']}** (score {c['score']:.2f})")
            st.caption(c["text"][:400] + "...")


store = get_store()
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", disabled=not files):
        with st.spinner("Indexing..."):
            for f in files:
                if f.name in store.sources:
                    continue
                store.add(load_pdf(f.getvalue(), f.name))
        st.success(f"Indexed: {', '.join(store.sources)}")
    if store.sources:
        st.caption("Loaded documents")
        for s in store.sources:
            st.write(f"• {s}")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if q := st.chat_input("Ask a question about your documents"):
    if not store.sources:
        st.warning("Upload and process at least one PDF first.")
        st.stop()
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = ask(q, st.session_state.history, store)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        st.markdown(res["answer"])
        if res["standalone"] != q:
            st.caption(f"Interpreted as: {res['standalone']}")
        if res["sources"]:
            show_sources(res["sources"])
    st.session_state.history.append({"role": "user", "content": q})
    st.session_state.history.append(
        {"role": "assistant", "content": res["answer"], "sources": res["sources"]}
    )