import streamlit as st
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
# --- Constants ---
FAISS_DB_PATH = "vectorstore/faiss_db"
EMBEDDING_MODEL_NAME = "intfloat/e5-small"
LLM_MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# --- Prompt Template ---
PROMPT_TEMPLATE = """
You are an AI tutor helping students prepare for the GATE CSE exam. Use the provided textbook-based context to answer the student's question.

### Instructions:
- ONLY use the context below to answer. If the answer is not present, say: *"Sorry, the context does not contain enough information to answer this question."*
- Be **accurate**, **concise**, and **easy to understand**.
- Use **bullet points** or **markdown tables** for comparisons or lists.
- Do NOT repeat the question in your response.
- Maintain a professional yet friendly tone suitable for GATE-level preparation.
- If a formula or concept is explained, ensure it’s **clear and beginner-friendly**.

---

### Context:
{context}

---

### Question:
{question}

---

### Answer:
"""


# --- Loaders ---
@st.cache_resource(show_spinner=False)
def get_retriever_and_llm():
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})
    if os.path.exists(EMBEDDING_MODEL_NAME):
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME,trust_remote_code = True)
    else:
        tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small",trust_remote_code = True)
    
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL_NAME,
        model_kwargs ={"device":"cpu"},
    )
    vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=2048,
        n_batch=64,
        n_threads=4,
        verbose=False
    )
    return retriever, llm

# --- Generator ---
def generate_answer(question, retriever, llm):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm(prompt, max_tokens=1024, stop=["Question:", "Context:"])
    return response["choices"][0]["text"].strip()

# --- Streamlit UI ---
st.set_page_config(page_title="GATE CSE Chat", layout="wide")
st.title("🎓 GATE CSE Chat Assistant")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

retriever, llm = get_retriever_and_llm()

# --- Chat Container ---
with st.container():
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

# --- User Input ---
if prompt := st.chat_input("Ask a question about GATE CSE..."):
    # User message
    with st.chat_message("user"):
        st.markdown(prompt)

    # LLM response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_answer(prompt, retriever, llm)
            st.markdown(response)

    # Store in history
    st.session_state.chat_history.append({"question": prompt, "answer": response})

# --- Sidebar: Reset Button ---
with st.sidebar:
    st.subheader("🧾 Chat Options")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
