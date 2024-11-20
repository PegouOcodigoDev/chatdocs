import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def save_to_temp_file(file):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with open(temp_file.name, "wb") as f:
            f.write(file.getvalue())
        return temp_file.name
    except Exception as e:
        print(f"Erro ao salvar arquivo temporário: {str(e)}")
        raise

def load_pdfs(uploads):
    docs = []
    for file in uploads:
        temp_file_path = save_to_temp_file(file)
        print(f"Arquivo salvo temporariamente em: {temp_file_path}")
        
        try:
            loader = PyPDFLoader(temp_file_path)
            loaded_docs = loader.load()
            if not loaded_docs:
                raise ValueError(f"O arquivo {file.name} não contém texto legível.")
            docs.extend(loaded_docs)
        finally:
            os.remove(temp_file_path)
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_docs = text_splitter.split_documents(docs)
    if not splitted_docs:
        raise ValueError("Nenhum documento foi gerado após a divisão.")
    return splitted_docs

def create_vector_store(splitted_docs, embedding_model="BAAI/bge-m3", save_path="vectorstore/db_faiss"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_documents(splitted_docs, embedding=embeddings)
    vector_store.save_local(save_path)
    return vector_store

def configure_retriever(vector_store, search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4}):
    return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

def config_retriever(uploads):
    try:
        docs = load_pdfs(uploads)
        splitted_docs = split_documents(docs)
        vector_store = create_vector_store(splitted_docs)
        retriever = configure_retriever(vector_store)
        return retriever
    except Exception as e:
        print(f"Erro ao configurar o retriever: {str(e)}")
        raise
