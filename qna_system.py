import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True,
)

def ask_question(question):
    print("Inside ask_question")
    result = qa_chain.invoke({"query": question})
    print("queried", result)
    
    # Format the response
    response = {
        "answer": result["result"],
        "sources": []
    }
    
    # Extract unique sources from the source documents
    seen_sources = set()
    for doc in result["source_documents"]:
        source_key = f"{doc.metadata['source_file']}-{doc.metadata['page_number']}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            response["sources"].append({
                "source_file": doc.metadata["source_file"],
                "page_number": doc.metadata["page_number"],
                "page_content": doc.page_content
            })
    
    return response