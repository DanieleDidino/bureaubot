from utils import get_filename_from_title
from engines_responses import default_engine, response_from_query_engine
from prompts import create_prompt_template

import streamlit as st
# import environ
import openai
from pathlib import Path
import pickle 


####################################################################################
# Config streamlit

# About text for the menu item
about = """
This chatbot uses documents from the Agentur für Arbeit as a data source.
It's important to note that this is a personal project utilizing publicly available documents and is NOT an official product of the Agentur für Arbeit.

Please be aware that the information provided by the chatbot may not always be accurate.
It is advisable to cross-verify any critical information from reliable sources before making any decisions based on the chatbot's response.

Page icon by https://icons8.com
"""

# streamlit config
st.set_page_config(
    page_title="Chatbot AfA",
    layout="wide",
    page_icon=".streamlit/icons8-chatbot-96.png",
    menu_items={
        "About": about
    }
)

st.header("Please enter your OpenAI Key")

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


####################################################################################
# Left column (first part)

sidebar = st.sidebar

with sidebar:

    # Custom page title and subtitle
    st.title("Bureau Bot")
    st.subheader("(Unofficial) Chatbot based on Agentur für Arbeit documents", divider="orange")
    st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown("<br>", unsafe_allow_html=True)

    # Get OpenAI ley from user
    openai_label = "Enter your [OpenAi key](https://platform.openai.com/account/api-keys)"
    OPENAI_KEY = st.text_input(label=openai_label, type="password", help="Enter your OpenAi key")

    # Add space between elements of the column
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize list downloadable files
    if "list_file_download" not in st.session_state:
        st.session_state.list_file_download = []
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


####################################################################################
# General Setup 

if OPENAI_KEY:
    # For importing a OpenAI key from a file, uncomment these line:
    # env = environ.Env()             # uncomment if key imported from file
    # environ.Env.read_env()          # uncomment if key imported from file
    # API_KEY = env("OPENAI_API_KEY") # uncomment if key imported from file
    # openai.api_key = API_KEY        # uncomment if key imported from file
    openai.api_key = OPENAI_KEY # comment if key imported from file

    # Define prompt
    qa_template = create_prompt_template()

    number_top_results = 5 # Number of top results to return
    folder_with_index = "vector_db" # load vector store from here

    # Load default query engine
    query_engine = default_engine(folder_with_index, qa_template, number_top_results)

# Load dictionary with the title of the pdf files.
with open(Path("pdf_titles", "pdf_dictionary.pkl"), 'rb') as f:
    pdf_dict = pickle.load(f)


####################################################################################
# Chat

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

    if OPENAI_KEY:
        response_for_user = response_from_query_engine(query_engine, prompt, pdf_dict)
    else:
        response_for_user = "Please add your OpenAI key to continue."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_for_user, unsafe_allow_html=True)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_for_user})


####################################################################################
# Left column (second part)

with sidebar:
    # Selectbox with the list of titles used as source
    selected_file = st.selectbox("Select a source file to download", options=st.session_state.list_file_download)
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
            mime='application/octet-stream')
