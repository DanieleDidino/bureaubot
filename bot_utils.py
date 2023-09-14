from llama_index import StorageContext, load_index_from_storage
from llama_index import Prompt
from llama_index import PromptHelper
from llama_index import SimpleDirectoryReader
from llama_index import LLMPredictor, ServiceContext
from llama_index import VectorStoreIndex
from llama_index.evaluation import ResponseEvaluator
from llama_index.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import streamlit as st


def default_engine(folder_with_index, qa_template, number_top_results):
    """
    Rebuild storage context from a vector database and return a query engine.

    Args:
        folder_with_index (str): Folder where the vector database is.
        qa_template (f-string): A prompt used to create the query engine.
        number_top_results (int): Number of top results to return

    Returns:
        query_engine: a query_engine created from the index.
    """
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=folder_with_index)
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=number_top_results)
    return query_engine


def query_engine_from_upload(folder_with_uploaded_file, qa_template, number_top_results):
    """
    Build storage context from uploaded documents and return an query_engine.

    Args:
        folder_with_uploaded_file (str): Folder with the files uploaded by the user.
        qa_template (f-string): A prompt used to create the query engine.
        number_top_results (int): Number of top results to return

    Returns:
        query_engine: A query_engine created from the index.
    """
    documents = SimpleDirectoryReader(input_dir=folder_with_uploaded_file).load_data()
    llm = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm)
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=number_top_results)
    return query_engine


def response_from_query_engine(query_engine, prompt, use_user_docs, uploaded_file, pdf_dict):
    """
    Return the response from the query engine.

    Args:
        query_engine: A query engine.
        prompt (str): Prompt from the user.
        use_user_docs (bool): It defines whether we use the default query engine (based on our documents)
                              or the query engine created from the user uploaded documents.
        uploaded_file (bool): It defines whether or not the user uploaded files. 
        pdf_dict (dict): Dictionary with the title of the pdf files (our documents).

    Returns:
        response_for_user (str): response produced by the LLM.
    """

    # Response from llm
    response = query_engine.query(prompt)
    # Evaluate the quality of the response
    eval_result = eval_response(response)
    # Get response as string
    response_text = response.response
    # If the user uploaded a file and switched to "query_user_engine"
    if use_user_docs and uploaded_file:
        if eval_result == "YES":
            response_for_user = response_text
        elif eval_result == "NO":
            response_for_user = "Sorry, the context information provided does not mention this information."
        else:
            response_for_user = "Something went wrong, try again!"
    # If we use our "query_engine_default"
    else:
        if eval_result == "YES":
            # Create first part of the source section (i.e., section of the response message with source documents)
            response_metadata_message = f'There are {len(response.metadata)} source files.'
            # Loop over all documents used as source
            for i, meta_data in enumerate(response.metadata):
                # Extract title from dictionary with {"file_name":"title"}, given a file name
                document_title = pdf_dict[response.metadata[meta_data]["file_name"]]
                # Append the title, if title is not in the list of used sources
                if document_title not in st.session_state.list_file_download:
                    st.session_state.list_file_download.append(document_title)
                # Update the source section with the source metadata
                response_metadata_message += f'<br>**Source {i+1}**: page {response.metadata[meta_data]["page_label"]} from file *{document_title}*.'
            # Add response_metadata_message (i.e., source section) after the LLM response text
            response_for_user = (f"{response_text}<br><br>{response_metadata_message}")
        elif eval_result == "NO":
            response_for_user = "Sorry, the context information provided does not mention this information."
        else:
            response_for_user = "Something went wrong, try again!"
    return response_for_user


def eval_response(response):
    llm= LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context_eval = ServiceContext.from_defaults(llm_predictor=llm)
    evaluator = ResponseEvaluator(service_context=service_context_eval)
    return evaluator.evaluate(response)
    

def default_chat_engine(folder_with_index, number_top_results):
    """
    Rebuild storage context from a vector database and return a chat engine.


    Args:
        folder_with_index (str): Folder where the vector database is.
        number_top_results (int): Number of top results to return

    Returns:
        query_engine: a query_engine created from the index.
    """
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=folder_with_index)
    # load index
    index = load_index_from_storage(storage_context)
    # Configure prompt parameters and initialise helper
    # max_input_size = 4096
    # num_output = 256
    # max_chunk_overlap = 0.2
    # prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    # system_prompt = (
    #     """
    #     You are an expert on the German administration system and your job is to answer technical questions.
    #     Assume that all questions are related to the the provided context.
    #     Keep your answers based on facts, do not hallucinate information.
    #     """
    # )
    # llm=LLMPredictor(llm=OpenAI(
    #     temperature=0,
    #     model_name="gpt-3.5-turbo",
    #     system_prompt=system_prompt
    # ))
    # service_context = ServiceContext.from_defaults(llm_predictor=llm, prompt_helper=prompt_helper)
    llm = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm)
    chat_engine = index.as_chat_engine(service_context=service_context, chat_mode="context", similarity_top_k=number_top_results)
    return chat_engine


def chat_engine_from_upload(folder_with_uploaded_file, number_top_results):
    """
    Build storage context from uploaded documents and return an chat_engine.

    Args:
        folder_with_uploaded_file (str): Folder with the files uploaded by the user.
        number_top_results (int): Number of top results to return.

    Returns:
        chat_engine: a chat_engine created from the index.
    """
    documents = SimpleDirectoryReader(input_dir=folder_with_uploaded_file).load_data()
    llm = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm)
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    chat_engine = index.as_chat_engine(service_context=service_context, similarity_top_k=number_top_results)
    return chat_engine


def response_from_chat_engine(chat_engine, prompt, use_user_docs, uploaded_file, pdf_dict):
    """
    Return the response from the chat engine.

    Args:
        chat_engine: A chat engine.
        prompt (str): Prompt from the user.
        use_user_docs (bool): It defines whether we use the default query engine (based on our documents)
                              or the query engine created from the user uploaded documents.
        uploaded_file (bool): It defines whether or not the user uploaded files. 
        pdf_dict (dict): Dictionary with the title of the pdf files (our documents).

    Returns:
        response_for_user (str): response produced by the LLM.
    """

    # Response from llm
    response = chat_engine.chat(prompt)
    # Evaluate the quality of the response
    eval_result = eval_response(response)
    # Get response as string
    response_text = response.response
    # If the user uploaded a file and switched to "query_user_engine"
    if use_user_docs and uploaded_file:
        if eval_result == "YES":
            response_for_user = response_text
        elif eval_result == "NO":
            response_for_user = "Sorry, the context information provided does not mention this information."
        else:
            response_for_user = "Something went wrong, try again!"
    # If we use our "query_engine_default"
    else:
        if eval_result == "YES":
            # Create first part of the source section (i.e., section of the response message with source documents)
            response_metadata_message = f'There are {len(response.source_nodes)} source files.'
            # Loop over all documents used as source
            for i, source_node in enumerate(response.source_nodes):
                # Extract title from dictionary with {"file_name":"title"}, given a file name
                document_title = pdf_dict[source_node.node.metadata["file_name"]]
                # Append the title, if title is not in the list of used sources
                if document_title not in st.session_state.list_file_download:
                    st.session_state.list_file_download.append(document_title)
                # Update the source section with the source metadata
                response_metadata_message += f'<br>**Source {i+1}**: page {source_node.node.metadata["page_label"]} from file *{document_title}*.'
            # Add response_metadata_message (i.e., source section) after the LLM response text
            response_for_user = (f"{response_text}<br><br>{response_metadata_message}")
        elif eval_result == "NO":
            response_for_user = "Sorry, the context information provided does not mention this information."
        else:
            response_for_user = "Something went wrong, try again!"
    return response_for_user


def save_uploadedfile(uploaded_file, folder_user):
    """
    Save the files uploaded by the user in the folder "folder_user".

    Args:
        uploaded_file: Files uploaded by the user.
        folder_user (str): Folder where to save the file(s).

    Returns:
        None
    """
    with open(os.path.join(folder_user, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return None


def get_filename_from_title(my_dict, title):
    """
    Find a file_name (i.e., the key) in the dictionary given a title (i.e., the value).
    The dictionary has the following structure:
    {
        "file_name_1":"title_1",
        "file_name_2":"title_2",
        ...
    }

    Args:
        my_dict (dict): The dictionary with file names and titles.
        title (str): The title we want to use to find the file name (i.e., the key).

    Returns:
        query_engine: a query_engine created from the index.
    """
    keys_list = list(my_dict.keys())
    values_list = list(my_dict.values())
    return keys_list[values_list.index(title)]
