import streamlit as st
from rag_qa import load_vectorstore, load_llm, generate_answer

FAISS_DB_PATH = "vectorstore/faiss_db"
LLM_MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
EMBEDDING_MODEL_NAME = "intfloat/e5-small"

@st.cache_resource(show_spinner=False)
def get_retriever_and_llm():
    vectorstore = load_vectorstore(FAISS_DB_PATH, EMBEDDING_MODEL_NAME)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_llm(LLM_MODEL_PATH)
    return retriever, llm

def main():
    st.set_page_config(page_title="GATE CSE Assistant", page_icon="🎓")
    st.markdown("<h1 style='text-align: center;'>🎓 GATE CSE Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Ask any technical question related to GATE Computer Science syllabus.</p>", unsafe_allow_html=True)

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar Chat History
    with st.sidebar:
        st.title("🧾 Chat History")
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                st.markdown(f"**Q{i}:** {chat['question']}")
                st.markdown(f"**A{i}:** {chat['answer']}")
        else:
            st.info("No questions asked yet.")
        
        # Clear history button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []

    # Load retriever and LLM
    retriever, llm = get_retriever_and_llm()

    # Main interaction
    question = st.text_input("💬 Type your question:", placeholder="e.g. Explain paging in OS.")
    if st.button("Get Answer") and question.strip():
        with st.spinner("Thinking..."):
            answer = generate_answer(question, retriever, llm)
            st.session_state.chat_history.append({"question": question, "answer": answer})
            st.markdown("### ✅ Answer")
            st.success(answer)

if __name__ == "__main__":
    main()
