from llama_index import StorageContext, load_index_from_storage
import streamlit as st


def default_engine(folder_with_index, qa_template, number_top_results):
    """
    Rebuild storage context from a vector database and return a query engine.

    Args:
        folder_with_index (str): Folder where the vector database is.
        qa_template (f-string): A prompt used to create the query engine.
        number_top_results (int): Number of top results to return.

    Returns:
        query_engine: a query_engine created from the index.
    """
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=folder_with_index)
    # load index
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=number_top_results)
    return query_engine


def response_from_query_engine(query_engine, prompt, pdf_dict):
    """
    Return the response from the query engine.

    Args:
        query_engine: A query engine.
        prompt (str): Prompt from the user.
        pdf_dict (dict): Dictionary with the title of the pdf files (our documents).

    Returns:
        response_for_user (str): response produced by the LLM.
    """
    # Response from llm
    response = query_engine.query(prompt)
    # Get response as string
    response_text = response.response
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
        score = response.source_nodes[i].score
        response_metadata_message += f'<br>**Source {i+1}**: page {response.metadata[meta_data]["page_label"]} from file *{document_title}*. Score: {score:.4f}.'
    # Add response_metadata_message (i.e., source section) after the LLM response text
    response_for_user = (f"{response_text}<br><br>{response_metadata_message}")
    return response_for_user
