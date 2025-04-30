# main.py
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from vector import load_csv_data, create_vector_store

# Load CSV data and create vector store
df = load_csv_data("./data/train_data.csv")
vector_store = create_vector_store(df)

# Initialize retriever and LLM
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = Ollama(model="llama3.2")

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("📚 CSV-Powered RAG Chatbot")
user_query = st.text_input("Ask a question based on your CSV data:")

if user_query:
    response = qa_chain.run(user_query)
    st.write("🤖", response)
