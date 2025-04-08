import streamlit as st
from document_processor import process_pdf, process_csv, generate_summary
from qna_system import ask_question
import os
from dotenv import load_dotenv
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

st.title("Document Question Answering System")

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'ready_for_qa' not in st.session_state:
    st.session_state.ready_for_qa = False

# File upload section
uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["pdf", "csv"])

async def process_file_async(file_path, file_type):
    try:
        st.session_state.processing = True
        st.session_state.ready_for_qa = False
        st.session_state.summary = None
        
        # Process document
        if file_type == "application/pdf":
            result = process_pdf(file_path)
        else:
            result = process_csv(file_path)
        
        st.session_state.ready_for_qa = True
        
        # # Generate summary in background
        # with ThreadPoolExecutor() as executor:
        #     future = executor.submit(generate_summary, result["documents"])
        st.session_state.summary = result["summary"]
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    finally:
        st.session_state.processing = False

if uploaded_file and not st.session_state.processing:
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Start async processing
    asyncio.run(process_file_async(file_path, uploaded_file.type))

st.subheader("Document Processing")

if st.session_state.processing:
    with st.status("Processing document...", expanded=True) as status:
        st.write("Breaking document into chunks...")
        if st.session_state.ready_for_qa:
            st.write("✅ Document ready for Q&A!")
            st.write("Generating summary...")
        status.update(label="Processing complete!", state="complete")
elif st.session_state.ready_for_qa:
    st.success("Document ready for Q&A!")

# Q&A Section
if st.session_state.ready_for_qa:
    question = st.text_input("Ask about the document")
    if question:
        try:
            response = ask_question(question)
            st.write("**Answer:**")
            st.write(response['answer'])
            
            with st.expander("Sources Used"):
                for source in response['sources']:
                    st.write(f"**{source['source_file']} (Page {source['page_number']})**")
                    st.write(source['page_content'])
        except Exception as e:
            st.error(f"Error answering question: {str(e)}")

st.subheader("Document Summary")

if st.session_state.summary:
    st.write(st.session_state.summary)
elif st.session_state.ready_for_qa:
    with st.spinner("Generating summary..."):
        while not st.session_state.summary:
            pass
        st.write(st.session_state.summary)
else:
    st.info("Summary will appear here when ready")