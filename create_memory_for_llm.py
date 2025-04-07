# from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# import os
# # Step 1: load raw PDF(s)
# DATA_PATH = "data/"
# FAISS_DB_PATH = "vectorstore/faiss_db"
# EMBEDDING_MODEL_NAME = "intfloat/e5-small"
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50

# def load_pdf_files(data_path):
#     loader = DirectoryLoader(
#         data_path,
#         glob = "*.pdf",
#         loader_cls = PyMuPDFLoader
#     )
#     documents = loader.load()
#     print(f"[INFO] Loaded {len(documents)} Pages from PDF Files.")
#     return documents

# def create_chunks(documents,chunk_size = CHUNK_SIZE,chunk_overlap = CHUNK_OVERLAP):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size = chunk_size,
#         chunk_overlap = chunk_overlap,
#     )
#     chunks = splitter.split_documents(documents=documents)
    
#     print(f"[INFO] Created {len(chunks)} Chunks.")
#     return chunks

# def format_e5_chunks(chunks):
#     for chunk in chunks:
#         chunk.page_content = "passage: "+ chunk.page_content
        
#     return chunks

# def get_embedding_model(model_name):
#     model_kwargs = {"device":"cpu"}
#     return HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs)

# def store_in_faiss(chunks, embedding_model, persist_path):
#     if not os.path.exists(persist_path):
#         os.makedirs(persist_path)

#     db = FAISS.from_documents(
#         documents=chunks,
#         embedding=embedding_model
#     )
#     db.save_local(persist_path)
#     print(f"[INFO] FAISS Vector DB saved at: {persist_path}")
    

# def main():
#     documents = load_pdf_files(DATA_PATH)
#     chunks = create_chunks(documents= documents)
#     formatted_chunks = format_e5_chunks(chunks)
#     embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
#     store_in_faiss(formatted_chunks,embedding_model,FAISS_DB_PATH)

# if __name__ == "__main__":
#     main()    