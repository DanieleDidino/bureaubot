import json
import logging
import os
import re
from typing import List

import openai
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.utilities import GoogleSearchAPIWrapper
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from llama_index import Prompt, StorageContext, load_index_from_storage
from pydantic import BaseModel, Field

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Search
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)


## Index Tool

# prompt for the index 
qa_template = Prompt(
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Do not give me an answer if it is not mentioned in the context as a fact.\n"
    "Always state where the information was found, and if possible where more information can be accessed.\n"
    "Given this information, please provide me with an answer to the following question:\n"
    "\n{query_str}\n"
)

# LLM
llm = ChatOpenAI(temperature=0)

## Index tool from documents in vector store
storage_context = StorageContext.from_defaults(persist_dir="vector_db")
index = load_index_from_storage(storage_context)

# engine from the index
query_engine = index.as_query_engine(
    streaming=False, text_qa_template=qa_template, similarity_top_k=5
)

# 
def parse_index_response(index_response):
    '''Filters the neccessary fields from the index response.

    Args:
        index_response (dict): dictionary with superflous fields

    Returns:
        dict: reduced fields -> 'output_text' & 'Documents'
    '''
    response_dict = {}
    response_dict["output_text"] = index_response.response
    response_dict["Documents"] = index_response.metadata
    return response_dict


# combined index query and parse into dict
def parsed_index_chain(user_input):
    '''Generates answer and cleans the output from the index.

    Args:
        user_input (str): user_question

    Returns:
        dict: dictionary with the neccessary fields: 'output_text' & 'Documents'
    '''
    raw_answer = query_engine.query(user_input)
    parsed_answer = parse_index_response(raw_answer)
    return parsed_answer


## Retriver Tool

# prompt to extend the google search
search_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Google search 
    results. Generate FIVE Google search queries that are similar to
    this question. The output should be a numbered list of questions and each
    should have a question mark at the end: {question}""",
)

class LineList(BaseModel):
    """List of questions."""

    lines: List[str] = Field(description="Questions")


class QuestionListOutputParser(PydanticOutputParser):
    """Output parser for a list of numbered questions."""

    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = re.findall(r"\d+\..*?\n", text)
        return LineList(lines=lines)

llm_chain = LLMChain(
    llm=llm,
    prompt=search_prompt,
    output_parser=QuestionListOutputParser(),
)

# Vectorstore
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
)

# Search: goes through programmable search engine
search = GoogleSearchAPIWrapper()

# Initialize the retriever 
web_research_retriever_llm_chain = WebResearchRetriever(
    vectorstore=vectorstore,
    llm_chain=llm_chain,
    search=search,
)

def parse_websearch_output(websearch_output):
    '''Creates a dictionary withe the neccessary fields from the websearch output.
    The created dictionary then resemnbles the aprsed output of the index tool for 
    so that we can read out the answers of both tools the same way further downstream.

    Args:
        websearch_output (tuple): (dict{'output_text':...}, list['Doucments':...])

    Returns:
        dict: {'output_text':..., 'Doucments':{'Document #', {'title': ... , 'content': ... 'source': ...}}
    '''
    wsearch_dict = {}
    wsearch_dict["output_text"] = websearch_output[0]["output_text"]
    wsearch_dict["Documents"] = {}
    # enumerate all documents and create sub-dictionaries
    for i, doc in enumerate(websearch_output[1]):
        document_info = {
            f"Document {i}": {
                "title": doc.metadata["title"],
                "content": doc.page_content,
                "source": doc.metadata["source"],
            }
        }

        # Append the new document to the existing documents dictionary
        wsearch_dict["Documents"].update(document_info)
    return wsearch_dict

def web_chain(user_input):
    """Collects documents and produces answer based on those returns answer + docs as a  tuple.
    
    Args:
        user_input (Str): user question to the agent

    Returns:
        tuple: dict{'output_text': answer}, list['Documents'...]
    """
    docs = web_research_retriever_llm_chain.get_relevant_documents(user_input)
    chain = load_qa_chain(llm, chain_type="stuff")
    return ( chain( {"input_documents": docs, "question": user_input}, return_only_outputs=True ), docs,
    )

def parse_web_chain(user_input):
    """Create structured output from the web retriever.
    
    Args:
        user_input (Str): user question to the agent

    Returns:
        dictionary: {'output_text':..., 'Documents': {'title': ... , 'content': ... , 'source': ...}, {}, {}, ...}
    """
    web_chain_response = web_chain(user_input)
    parsed_output = parse_websearch_output(web_chain_response)
    return parsed_output

tools = [
    Tool(
        name="LlamaIndex",
        func=lambda q: json.dumps(parsed_index_chain(q)),
        description="Useful for when you want to answer questions about work and unemployment.",
        return_direct=True,
    ),
    Tool(
        name="Websearch",
        func=lambda q: json.dumps(parse_web_chain(q)),
        description="Useful for when you want to answer more general questions concerning laws and administration in Berlin, Germany.",
        return_direct=True,
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history")

agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

result = agent_executor.run(
    # input="Where can I find information about the duties of landlords?"
    input="What do I have to do if I know I will be unemployed by next month?"
)

def print_agent_output(agent_output):
    '''Converts the string back into a dict and prints the structured output.

    Args:
        agent_output (str): agent returns a string of the dictionary
    
    Return: 
        Prints the dictionary.
    '''
    agent_output_dict = json.loads(agent_output)
    print(agent_output_dict['output_text'])

    for doc, dict  in agent_output_dict['Documents'].items():
        print(doc)
        for k, v in dict.items():
            print(f"{k}: {v}\n")
    
    
def complete_agent_chain(user_question):
    '''Get's an answer from the agent and prints the structrued output.

    Args:
        user_question (str): _description_

    Returns:
        print: _description_
    '''
    raw_agent_response = agent_executor.run(input=user_question)
    # response_dict = json.loads(raw_agent_response)
    # print_agent_output(response_dict)
    return raw_agent_response