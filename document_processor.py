import os

from dotenv import load_dotenv
from queue import Queue
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

from summary import generate_summary
from search_index import index_schema

load_dotenv()

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize Azure search client
search_client = SearchIndexClient(
    endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
)

# Check if index exists, create if not
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
if index_name not in [index.name for index in search_client.list_indexes()]:
    # Create index using schema definition above
    search_client.create_or_update_index(index_schema)

print(os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"), os.getenv("AZURE_EMBEDDING_DEPLOYMENT"), os.getenv("AZURE_SEARCH_INDEX_NAME"))
# Initialize Azure AI Search vector store
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    embedding_function=embeddings.embed_query,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

def process_pdf(file_path, progress_queue=None):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split(text_splitter)
    
    # Add source metadata
    for i, page in enumerate(pages):
        page.metadata["source_file"] = os.path.basename(file_path)
        page.metadata["page_number"] = i + 1
    
    # Convert to LangChain documents
    documents = [
        Document(
            page_content=page.page_content,
            metadata={
                "page_number":page.metadata["page_number"],
                "source_file":page.metadata["source_file"]
            }
            
        ) for page in pages
    ]
    
    if progress_queue:
        progress_queue.put({"stage": "chunking", "progress": 50})
        
    # Stage 3: Summary generation (runs in parallel)
    # def generate_summary_async():
    #     summary = generate_summary(pages)
    #     if progress_queue:
    #         progress_queue.put({"stage": "summary", "progress": 100, "summary": summary})
            
    # summary_thread = threading.Thread(target=generate_summary_async, args=(documents,))
    # summary_thread.start()
    
    summary = generate_summary(pages)
    
    vector_store.add_documents(documents)
    
    return {
        "num_chunks": len(documents),
        "ready_for_qa": True,
        "summary": summary
    }

def process_csv(file_path):
    loader = CSVLoader(file_path)
    documents = loader.load_and_split(text_splitter)
    
    # Add source metadata
    for i, doc in enumerate(documents):
        doc.metadata["source_file"] = os.path.basename(file_path)
        doc.metadata["page_number"] = i + 1  # Using row number as "page"
    
    vector_store.add_documents(documents)
    return len(documents)