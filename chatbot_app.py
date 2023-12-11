import pickle

# import os
from pathlib import Path

import environ
import openai
import streamlit as st
from bot_utils import (
    chat_engine_from_upload,
    default_chat_engine,
    default_engine,
    get_filename_from_title,
    query_engine_from_upload,
    response_from_chat_engine,
    response_from_query_engine,
    save_uploadedfile,
)

# from streamlit_extras.app_logo import add_logo
from langchain.chat_models import ChatOpenAI

# from llama_index import StorageContext, load_index_from_storage
from llama_index import Prompt

# from PIL import Image
from streamlit_chat import message

from tool_agent import complete_agent_chain

# TODO: For now I use my key, then use the user key
env = environ.Env()
environ.Env.read_env()
API_KEY = env("OPENAI_API_KEY")
openai.api_key = API_KEY


# Set LLm
selected_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Define prompt
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Do not give me an answer if it is not mentioned in the context as a fact. \n"
    "Given this information, please provide me with an answer to the following question:\n{query_str}\n"
)
qa_template = Prompt(template)

number_top_results = 5  # Number of top results to return

# Load files and set folder names

# Load dictionary with the title of the pdf files.
with open(Path("pdf_titles", "pdf_dictionary.pkl"), "rb") as f:
    pdf_dict = pickle.load(f)

folder_user_uploaded_files = "data_streamlit"

# Config streamlit

# streamlit config
st.set_page_config(
    page_title="Bureau Bot",
    layout="wide",
    page_icon=".streamlit/favicon.ico",
    menu_items={
        "Get Help": "https://www.bürohengst.com/help",
        "Report a bug": "https://www.bürohengst.com/bug",
        "About": "# Bürohengst makes german bureaucracy a joy ride. \
        Our robot paper pusher (Chatbot) knows a host of german official \
        documents (they're in a Vector database) and answers your questions about them.",
    },
)

# Hide the menu button
st.markdown(
    """ <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """,
    unsafe_allow_html=True,
)

# Condense the layout
padding = 0
st.markdown(
    f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """,
    unsafe_allow_html=True,
)

# load custom css styles
with open(".streamlit/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Left column (first part)
sidebar = st.sidebar

with sidebar:
    # Custom page title and subtitle
    st.title("Bureau Bot ")
    st.subheader("Ate the official documents", divider="orange")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Get OpenAI ley from user
    openai_label = (
        "Enter your [OpenAi key](https://platform.openai.com/account/api-keys)"
    )
    OPENAI_KEY = st.text_input(
        label=openai_label, type="password", help="Enter your OpenAi key"
    )
    # openai.api_key = OPENAI_KEY # TODO: Uncomment this lines when we will ask for the user OpenAI Key

    # This toggle define whether we use the default query engine (based on our documents) or
    # the query engine created from the user uploaded documents
    toggle_help = 'If "on" the responses are based on the uploaded file(s) only.'
    use_user_docs = st.toggle("User document(s)", key="my_toggle", help=toggle_help)
    st.write(
        f"User context is {use_user_docs}"
    )  # TODO: This info is only for us, delete before demo day

    # Expander for file uploading
    with st.expander("Choose a file from your hard drive"):
        uploaded_file = st.file_uploader(
            "", type=["docx", "doc", "pdf"], accept_multiple_files=True
        )
        if uploaded_file:
            st.text("File saved successfully!")

    # Add space between elements of the column
    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize list downloadable files
    if "list_file_download" not in st.session_state:
        st.session_state.list_file_download = []

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Save the uploaded file
if uploaded_file:
    for file in uploaded_file:
        save_uploadedfile(file, folder_user_uploaded_files)

# Chat

# check if files were uploaded
default = 1
if use_user_docs:
    if uploaded_file:
        upload_engine = query_engine_from_upload(
            folder_user_uploaded_files, qa_template, number_top_results, selected_llm
        )
        default = 0
    elif not uploaded_file:
        default = 1
else:
    default = 1

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("How may I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # choose toolchain or engine from uploads for the chat
    if default == 1:
        response_for_user = complete_agent_chain(prompt)
    else:
        response_for_user = response_from_query_engine(
            upload_engine, prompt, use_user_docs, uploaded_file, pdf_dict, selected_llm
        )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_for_user, unsafe_allow_html=True)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response_for_user}
    )



####################################################################################
# Left column (second part)

with sidebar:
    # Selectbox with the list of titles used as source
    selected_file = st.selectbox(
        "Select a source file to download", options=st.session_state.list_file_download
    )
    # Prepare the file for downloading, if a file is selected in the selectbox
    if selected_file:
        # Get the name of the file from the selected title
        file_name_to_download = get_filename_from_title(pdf_dict, selected_file)
        # Define path to file to download
        path_file_download = Path("documents_pdf", file_name_to_download)
        # Open and read the file
        with open(path_file_download, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        # Make the file available for download
        st.download_button(
            label="Download",
            data=PDFbyte,
            file_name=file_name_to_download,
            mime="application/octet-stream",
        )
