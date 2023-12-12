from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index import StorageContext
from llama_index import PromptHelper
from llama_index.llms import OpenAI
from llama_index import OpenAIEmbedding

import openai
import environ

env = environ.Env()
environ.Env.read_env()
API_KEY = env('OPENAI_API_KEY')
openai.api_key = API_KEY

doc_path = "documents_pdf" # where the documents are
embedding_path = "vector_db" # where the vector database is saved

def ask_overwrite():
    prompt_string = "Create a new vector database? This will OVERWRITE the old vector database. (Y/N)"
    overwrite = input(prompt_string)
    return overwrite == "Y"


def load_docs(doc_path):
    docs = SimpleDirectoryReader(input_dir=doc_path).load_data()
    return docs


def create_vector_db(docs):

    # ----------------------------------------------------------------------------------
    # # Define The ServiceContext: a bundle of commonly used resources used during
    # the indexing and querying stage in a LlamaIndex pipeline/application.

    llm= OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    
    # Configure prompt parameters and initialise helper
    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 0.1
    prompt_helper = PromptHelper(
          context_window=max_input_size,
          num_output=num_output,
          chunk_overlap_ratio=max_chunk_overlap,
          chunk_size_limit=None
        )
    
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        prompt_helper=prompt_helper)
    
    # ----------------------------------------------------------------------------------
    # Storage context: The storage context container is a utility container for storing 
    # nodes, indices, and vectors. It contains the following:
    # - docstore: BaseDocumentStore
    # - index_store: BaseIndexStore
    # - vector_store: VectorStore
    # - graph_store: GraphStore
    storage_context = StorageContext.from_defaults()

   # ----------------------------------------------------------------------------------
   # VectorStoreIndex: a data structure that allows for the retrieval of relevant context
   # for a user query. This is particularly useful for retrieval-augmented generation (RAG) use-cases.
   # VectorStoreIndex stores data in Node objects, which represent chunks of the original documents,
   # and exposes a Retriever interface that supports additional configuration and automation.
    print("Creating Vector Database ...")
    index = VectorStoreIndex.from_documents(
        docs,
        service_context=service_context,
        storage_context=storage_context
    )
    print("Done")

    return index


def save_vector_db(index, embedding_path):
    index.storage_context.persist(persist_dir=embedding_path)


if __name__ == "__main__":
    overwrite = ask_overwrite()

    if overwrite:
        docs = load_docs(doc_path)
        index = create_vector_db(docs)
        save_vector_db(index, embedding_path)
