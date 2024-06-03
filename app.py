import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)

# Function to read PDF
def read_pdf(pdf):
    raw_text = ""
    file = PdfReader(pdf)
    for page in file.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Function to process text and create a FAISS index
def process_text(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        separators=["\n"]
    )
    texts = text_splitter.create_documents([raw_text])

    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db = FAISS.from_documents(texts, embeddings)
    return db

# Function to initialize the QA chain
def initialize_chain(db):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    return chain

# Main function to run the app
def main():
    st.title("PDF Question Answering")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        raw_text = read_pdf(uploaded_file)
        db = process_text(raw_text)
        chain = initialize_chain(db)
        
        question = st.text_input("Ask a question about the PDF")
        if question:
            answer = chain.invoke(question)
            st.write(f"Answer: {answer['result']}")

if __name__ == "__main__":
    main()
