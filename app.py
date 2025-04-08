import streamlit as st
from document_processor import process_pdf, process_csv
from qna_system import ask_question
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

st.title("Document Question Answering System (LangChain Version)")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

if uploaded_file is not None:
    # Save the file temporarily
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Process the file
        if uploaded_file.type == "application/pdf":
            num_chunks = process_pdf(file_path)
        elif uploaded_file.type == "text/csv":
            num_chunks = process_csv(file_path)
        print("Uploaded successfully")
        st.success(f"Processed and uploaded {num_chunks} document chunks from {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Question answering section
question = st.text_input("Ask a question about the uploaded documents")

if question:
    try:
        response = ask_question(question)
        
        st.subheader("Answer")
        st.write(response['answer'])
        
        st.subheader("Sources Used")
        for source in response['sources']:
            with st.expander(f"Source: {source['source_file']} (Page {source['page_number']})"):
                st.write(source['page_content'])
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")