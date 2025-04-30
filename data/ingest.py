# ingest.py
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document

def load_and_prepare_documents(file_path):
    df = pd.read_csv(file_path)
    # Merge all columns into a single text field per row
    texts = df.astype(str).agg(" ".join, axis=1).tolist()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents(texts)
    return docs

def create_chroma_vectorstore(docs, persist_directory="chroma_db"):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)
    db.persist()
    print(f" Chroma vector store saved at: {persist_directory}")

if __name__ == "__main__":
    docs = load_and_prepare_documents("data.csv")
    create_chroma_vectorstore(docs)
