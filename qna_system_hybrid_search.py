import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import AzureSearchHybridRetriever
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize Azure AI Search vector store
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    embedding_function=embeddings.embed_query,
)

# Initialize Search Client for hybrid search
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Custom prompt to ensure source citations
prompt_template = """Use the following pieces of context to answer the question at the end. 
For each fact in your answer, include the source (filename and page number) in square brackets like this: [source.pdf, p.12].

{context}

Question: {question}
Answer with sources:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

# Use our hybrid retriever instead of the vector_store retriever
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=hybrid_retriever,  # Using the hybrid retriever
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True,
)

def hybrid_search(query_text, top_k=5):
    """
    Performs hybrid search combining vector and keyword search
    
    Args:
        query_text: The user's natural language query
        top_k: Number of results to return
    
    Returns:
        List of search results with their scores
    """
    # Get the vector embedding for the query
    query_vector = embeddings.embed_query(query_text)
    
    # Initialize the search client for searching (not index management)
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
    )
    
    # Execute hybrid search
    results = search_client.search(
        search_text=query_text,  # For keyword/BM25 search
        vector=query_vector,     # For vector search
        vector_fields="content_vector",
        top=top_k,
        query_type="semantic",   # Enable semantic ranking
        semantic_configuration_name="default",
        query_caption="extractive",
        query_answer="extractive",
        search_fields=["page_content"],
        select=["id", "page_content", "source_file", "page_number"],
        # You can adjust these weights to favor vector or keyword search
        vector_search_configuration={
            "name": "default",
            "weight": 0.7  # Weight for vector search (0.7 = 70% vector, 30% keyword)
        }
    )
    
    # Process and return results
    search_results = []
    for result in results:
        search_results.append({
            "id": result["id"],
            "content": result["page_content"],
            "source": result["source_file"],
            "page": result["page_number"],
            "score": result["@search.score"],
            "captions": result.get("@search.captions", [])
        })
    
    return search_results

def ask_question(question):
    result = qa_chain.invoke({"question": question})
    
    # Format the response
    response = {
        "answer": result["answer"],
        "sources": []
    }
    
    # Extract unique sources from the source documents
    seen_sources = set()
    for doc in result["source_documents"]:
        source_key = f"{doc.metadata['source_file']}-{doc.metadata['page_number']}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            response["sources"].append({
                "source": doc.metadata["source_file"],
                "page": doc.metadata["page_number"],
                "content": doc.page_content
            })
    
    return response