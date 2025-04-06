from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from llama_cpp import Llama
import os

FAISS_DB_PATH = "vectorstore/faiss_db"
EMBEDDING_MODEL_NAME = "intfloat/e5-small"
LLM_MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

def load_vectorstore(path,embedding_model_name):
    embeddings = HuggingFaceEmbeddings(
        model_name = embedding_model_name,
        model_kwargs = {"device":"cpu"}
    )
    
    vectorstore = FAISS.load_local(path,embeddings,allow_dangerous_deserialization=True)
    return vectorstore

def load_llm(model_path):
    return Llama(
        model_path = model_path,
        n_ctx = 2048,
        n_batch = 64,
        n_threads = 4,
        verbose = False
        )

def generate_answer(question, retriever, llm):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = llm(prompt, max_tokens=512, stop=["Question:", "Context:"])
    return response["choices"][0]["text"].strip()

    
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



def main():
    
    vectorstore = load_vectorstore(FAISS_DB_PATH,EMBEDDING_MODEL_NAME)
    retriever = vectorstore.as_retriever(search_kwargs = {"k":5})
    llm = load_llm(LLM_MODEL_PATH)
    
    question = input("Ask your GATE-CSE related question: ")
    
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    prompt = PROMPT_TEMPLATE.format(context = context,question = question)
    
    
    response = llm(prompt,max_tokens = 512,stop =   ["Question:","Context:"])
    print("\n--- Answer ---\n")
    print(response["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()