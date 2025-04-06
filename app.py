import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from llama_cpp import Llama



import os
os.environ["PORT"] = os.environ.get("PORT", "8501")

FAISS_DB_PATH = "vectorstore/faiss_db"
EMBEDDING_MODEL_NAME = "intfloat/e5-small"
LLM_MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Prompt Template
PROMPT_TEMPLATE = """
You are an AI tutor helping students prepare for the GATE CSE exam. Use the provided textbook-based context to answer the question accurately.

Instructions:
- Base your answer only on the context provided below. If the answer is not in the context, say "The context does not provide enough information to answer this question."
- For **comparative questions**, respond using a **markdown table** with clearly labeled headers and rows.
- Be **concise**, **accurate**, and **easy to understand**.
- Avoid repeating the question or instructions in the answer.
- Use bullet points or tables if that improves clarity.

Context:
{context}

Question:
{question}
"""

# Load FAISS Vectorstore
def load_vectorstore(path, embedding_model_name):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# Load LLaMA Model
def load_llm(model_path):
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_batch=64,
        n_threads=4,
        verbose=False
    )

# Generate Answer
def generate_answer(question, retriever, llm):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm(prompt, max_tokens=512, stop=["Question:", "Context:"])
    return response["choices"][0]["text"].strip()

# Cache retriever and llm
@st.cache_resource(show_spinner=False)
def get_retriever_and_llm():
    vectorstore = load_vectorstore(FAISS_DB_PATH, EMBEDDING_MODEL_NAME)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_llm(LLM_MODEL_PATH)
    return retriever, llm

# Streamlit App
def main():
    st.set_page_config(page_title="GATE CSE Assistant", page_icon="🎓")
    st.markdown("<h1 style='text-align: center;'>🎓 GATE CSE Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Ask any technical question related to GATE Computer Science syllabus.</p>", unsafe_allow_html=True)

    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.title("🧾 Chat History")
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                st.markdown(f"**Q{i}:** {chat['question']}")
                st.markdown(f"**A{i}:** {chat['answer']}")
        else:
            st.info("No questions asked yet.")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []

    # LLM and retriever
    retriever, llm = get_retriever_and_llm()

    # Question Input
    question = st.text_input("💬 Type your question:", placeholder="e.g. Explain paging in OS.")
    if st.button("Get Answer") and question.strip():
        with st.spinner("Thinking..."):
            answer = generate_answer(question, retriever, llm)
            st.session_state.chat_history.append({"question": question, "answer": answer})
            st.markdown("### ✅ Answer")
            st.success(answer)

if __name__ == "__main__":
    main()
