import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Add this function to your document_processor.py
def generate_summary(docs):
    # Map prompt
    map_template = """The following is a document excerpt:
    {text}
    
    Based on this, write a concise summary of this section:
    """
    map_prompt = PromptTemplate.from_template(map_template)
    
    # Reduce prompt
    reduce_template = """The following are summaries of document sections:
    {summaries}
    
    Combine these into one coherent summary of the entire document. 
    Focus on key themes, findings, and conclusions:
    """
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    
    # LLM chain for map step
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
    
    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    
    # 1. Map step: Process each document individually
    summaries = [map_chain.invoke({"text": doc.page_content}) for doc in docs]
    
    # 2. Reduce step: Combine all summaries
    final_summary = reduce_chain.invoke({"summaries": "\n\n".join(summaries)})
    
    return final_summary