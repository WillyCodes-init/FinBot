# vector.py
import pandas as pd
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_vector_store(df):
    # Combine 'Question' and 'Answer' columns into a single text field
    df['combined'] = df['question'] + " " + df['answer']
    documents = df['combined'].tolist()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(documents)

    # Generate embeddings
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

    # Create ChromaDB vector store
    vector_store = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory="./chroma_db")
    vector_store.persist()
    return vector_store
