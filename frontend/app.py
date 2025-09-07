import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Q&A (HF+FAISS)", page_icon="📚")
st.title("📚 RAG Q&A (Hugging Face + FAISS)")
st.caption("Upload a single document (PDF/Markdown/HTML) and ask questions. No API keys needed.")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "md", "markdown", "html", "htm"])
if uploaded_file is not None:
    with st.spinner("Uploading & indexing ..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            res = requests.post(f"{BACKEND_URL}/upload/", files=files, timeout=300)
            if res.status_code == 200:
                st.success(res.json().get("message", "File processed successfully"))
            else:
                st.error(f"Upload failed: {res.text}")
        except Exception as e:
            st.error(f"Upload error: {e}")

st.divider()

question = st.text_input("Ask a question about the uploaded document")
if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking ..."):
            try:
                res = requests.post(f"{BACKEND_URL}/ask/", data={"question": question}, timeout=300)
                if res.status_code == 200:
                    data = res.json()
                    st.markdown(f"**Answer:** {data.get('answer','(no answer)')}")
                    cits = data.get("citations", [])
                    if cits:
                        st.markdown("**Citations (top chunks):**")
                        for c in cits:
                            st.write(f"- {c.get('source')} (chunk {c.get('chunk_id')}) • score={round(c.get('score',0.0), 3)}")
                else:
                    st.error(f"Ask failed: {res.text}")
            except Exception as e:
                st.error(f"Ask error: {e}")
