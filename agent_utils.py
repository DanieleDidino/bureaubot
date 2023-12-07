import logging

from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from llama_index import StorageContext, load_index_from_storage

logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)




qa_template = PromptTemplate(
    """You are a helpful administrative assistant that help the user with legalese and bureaucracy.
    Only give answers based on facts which sources you can provide.
    Provide the sources that you're answers are based on.
    
    Answer this question 
    ---------------------\n
    {query_str} 
    ---------------------\n
    
    based only on the following context:
    ---------------------\n
    {context}
    ---------------------\n
    
    Give an informative and pragmatic answer.
    ---------------------\n
    Answer:
    ---------------------\n
    
    State the sources of the context information.
    ---------------------\n
    Sources: {sources}
    """
)

# LLM
llm = ChatOpenAI(temperature=0)

## Index tool from documents in vector store
storage_context = StorageContext.from_defaults(persist_dir="vector_db")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
    streaming=True, text_qa_template=qa_template, similarity_top_k=5
)