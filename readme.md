# Document Q&A and Summarization App

A Streamlit application that allows users to upload PDF documents, generate comprehensive summaries, and ask questions about the document content using Azure OpenAI services.

## Features

- **Document Upload**: Support for PDF documents
- **Document Summarization**: Generate concise summaries of uploaded documents
- **Question Answering**: Ask specific questions about document content
- **Azure OpenAI Integration**: Leverages Azure's OpenAI services for advanced natural language processing

## Installation

### Prerequisites

- Python 3.8+
- Azure OpenAI API access

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Azure OpenAI credentials:
   ```
    AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
    OPENAI_API_KEY=your-api-key
    AZURE_ENDPOINT=your-azure-endpoint
    AZURE_OPENAI_API_VERSION=your-api-version
    AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
    AZURE_SEARCH_SERVICE_ENDPOINT=your-azure-endpoint
    AZURE_SEARCH_API_KEY=your-azure-search-key
    AZURE_SEARCH_INDEX_NAME=your-azure-search-index-name
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically `http://localhost:8501`).

3. Upload a PDF document using the file uploader.

4. Use the sidebar to select between:
   - **Summarize**: Generate a concise summary of the document
   - **Ask Questions**: Ask specific questions about the document content

## Project Structure

```
document-qa-app/
├── app.py                  # Main Streamlit application
├── document_processor.py   # Document processing utilities
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (not tracked by git)
└── README.md               # Project documentation
```

## Key Components

### Document Processing

The application uses LangChain to process documents:

```python
# Example of the document processing flow
loader = PyPDFLoader(file_path)
pages = loader.load_and_split(text_splitter)
```

### Text Summarization

Document summarization is handled using a map-reduce approach:

```python
def generate_summary(docs):
    # Map step: Process each document individually
    summaries = [map_chain.invoke({"text": format_doc(doc)}) for doc in docs]
    
    # Reduce step: Combine all summaries
    final_summary = reduce_chain.invoke({"summaries": "\n\n".join(summaries)})
    
    return final_summary
```

### Question Answering

The Q&A functionality retrieves relevant document sections and formulates responses:

```python
# Example of Q&A functionality
response = qa_chain.invoke({"question": user_question, "docs": docs})
```

## Customization

### Modifying the Text Splitter

You can adjust the text splitter parameters in `document_processor.py` to change how documents are chunked:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
```

### Changing Azure OpenAI Parameters

You can modify the LLM parameters in the appropriate functions:

```python
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    temperature=0,  # Adjust for more creative (higher) or deterministic (lower) responses
    max_tokens=None,
    timeout=None,
    max_retries=2
)
```

## Troubleshooting

### Common Issues

1. **Empty Summaries**: If you receive generic or empty summaries, check that the document content is being properly extracted and formatted.

2. **API Connection Errors**: Verify your Azure OpenAI credentials in the `.env` file and check your internet connection.

3. **Memory Issues**: For large documents, consider adjusting the chunk size in the text splitter to process smaller segments of text.

## Dependencies

- `streamlit`: Web application framework
- `langchain`: Framework for LLM applications
- `langchain_openai`: OpenAI integration for LangChain
- `pypdf`: PDF processing
- `python-dotenv`: Environment variable management
