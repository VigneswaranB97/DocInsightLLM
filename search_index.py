import os
from dotenv import load_dotenv
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)

load_dotenv()

# Define the search index schema with hybrid search capabilities
index_schema = SearchIndex(
    name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    fields=[
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SearchableField(
            name="content", 
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",  # Use linguistic analyzer for better text processing
            sortable=True, 
            filterable=True, 
            facetable=True
        ),
        SearchableField(
            name="metadata", 
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
            sortable=True, 
            filterable=True, 
            facetable=True
        ),
        SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=3072,  # Adjust based on your embedding model dimensions
            vector_search_profile_name="my-vector-config",
            retrievable=True,
        )
    ],
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
    )
)