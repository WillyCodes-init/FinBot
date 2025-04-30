# app.py
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Load vector store
persist_dir = "chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Set up retrieval and LLM
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = Ollama(model="llama3")  # or another model you pulled
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("🔍 RAG Chatbot – Ask from CSV")
query = st.text_input("Ask a question:")
if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
        st.write("🤖", answer)
