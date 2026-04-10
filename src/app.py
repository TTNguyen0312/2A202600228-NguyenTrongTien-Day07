"""
Streamlit UI for the Education Policy RAG Agent.

Run with:
    python -m streamlit run src/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv

from main import INDEX_PATH, build_embedder, build_index, build_llm_fn, load_index
from src.agent import KnowledgeBaseAgent
from src.embeddings import EMBEDDING_PROVIDER_ENV

load_dotenv(override=False)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tra cứu Quy chế thi THPT",
    page_icon="📚",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
PHASE_LABELS = {
    "general": "Quy định chung",
    "preparation": "Chuẩn bị thi",
    "registration": "Đăng ký dự thi",
    "exam": "Coi thi",
    "grading": "Chấm thi",
    "appeals": "Phúc khảo",
}
CATEGORY_LABELS = {
    "quy_dinh_chung": "Quy định chung",
    "to_chuc_thi": "Tổ chức thi",
    "dang_ky_thi": "Đăng ký thi",
    "thi_sinh": "Thí sinh",
    "de_thi": "Đề thi",
    "coi_thi": "Coi thi",
    "cham_thi": "Chấm thi",
    "phuc_khao": "Phúc khảo",
}


# ── Chunk card renderer (native Streamlit, no HTML) ───────────────────────────
def render_chunk_cards(chunks: list[dict]) -> None:
    for i, chunk in enumerate(chunks, 1):
        meta   = chunk["metadata"]
        score  = chunk["score"]
        doc_id = str(meta.get("doc_id", meta.get("source", "?")))
        c_idx  = meta.get("chunk_index", "?")
        phase  = PHASE_LABELS.get(meta.get("phase", ""), meta.get("phase", "—"))
        cat    = CATEGORY_LABELS.get(meta.get("category", ""), meta.get("category", "—"))
        text   = chunk["content"]

        with st.container(border=True):
            # Header row: rank + doc id + chunk index
            left, right = st.columns([3, 1])
            with left:
                score_icon = "🟢" if score >= 0.65 else "🟡" if score >= 0.35 else "🔴"
                st.markdown(f"{score_icon} &nbsp; **#{i} · {doc_id}** &nbsp; `chunk {c_idx}`")
            with right:
                st.markdown(f"**Score: `{score:.4f}`**")

            # Score progress bar
            st.progress(float(max(0.0, min(1.0, score))))

            # Metadata tags
            t1, t2, t3 = st.columns(3)
            t1.caption(f"⏱ Phase: **{phase}**")
            t2.caption(f"📂 Category: **{cat}**")
            t3.caption(f"📄 Doc: `{doc_id}`")

            st.divider()

            # Chunk text — st.text prevents any markdown/HTML interpretation
            st.text(text)


# ── Load agent (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải index vào bộ nhớ...")
def get_agent():
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    embedder = build_embedder(provider)
    store    = load_index(INDEX_PATH, embedder)
    llm_fn   = build_llm_fn(provider)
    agent    = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    return store, agent, llm_fn


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    top_k = st.slider("Số chunks tham chiếu (top-k)", min_value=1, max_value=8, value=3)

    st.divider()
    st.subheader("Lọc metadata (tuỳ chọn)")

    phase_options    = ["(tất cả)", "general", "preparation", "registration", "exam", "grading", "appeals"]
    category_options = ["(tất cả)", "quy_dinh_chung", "to_chuc_thi", "dang_ky_thi",
                        "thi_sinh", "de_thi", "coi_thi", "cham_thi", "phuc_khao"]

    selected_phase    = st.selectbox("Phase",    phase_options,    format_func=lambda x: PHASE_LABELS.get(x, x))
    selected_category = st.selectbox("Category", category_options, format_func=lambda x: CATEGORY_LABELS.get(x, x))

    st.divider()
    st.subheader("🗂️ Index")
    index_exists = Path(INDEX_PATH).exists()
    st.success("Index sẵn sàng") if index_exists else st.warning("Chưa có index.")

    if st.button("🔨 Build Index", disabled=index_exists):
        with st.spinner("Đang chunk và embed tài liệu..."):
            try:
                build_index()
                st.success("Build xong!")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi: {e}")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📚 Tra cứu Quy chế thi tốt nghiệp THPT")
st.caption("Nguồn: Thông tư 15/2020/TT-BGDĐT · RAG powered by vector search")

if not Path(INDEX_PATH).exists():
    st.info("Nhấn **Build Index** ở sidebar để bắt đầu.")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "chunks" in msg:
            with st.expander(f"📄 {len(msg['chunks'])} chunks tham khảo"):
                render_chunk_cards(msg["chunks"])

# ── Input ─────────────────────────────────────────────────────────────────────
query = st.chat_input("Đặt câu hỏi về quy chế thi THPT...")

if query:
    metadata_filter: dict = {}
    if selected_phase != "(tất cả)":
        metadata_filter["phase"] = selected_phase
    if selected_category != "(tất cả)":
        metadata_filter["category"] = selected_category

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm chunks liên quan..."):
            store, agent, llm_fn = get_agent()
            chunks = (store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)
                      if metadata_filter else store.search(query, top_k=top_k))

        with st.spinner("Đang tạo câu trả lời..."):
            if metadata_filter:
                context = "\n\n".join(c["content"] for c in chunks)
                answer  = llm_fn(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
            else:
                answer = agent.answer(query, top_k=top_k)

        st.markdown(answer)

        if metadata_filter:
            st.caption(f"🔍 Filter: {metadata_filter}")

        with st.expander(f"📄 {len(chunks)} chunks tham khảo"):
            render_chunk_cards(chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
