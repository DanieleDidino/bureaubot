import logging
import os
import pdb
import re

import environ
import nest_asyncio
from bot_utils_marco import default_engine
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import (
    LLMChain,
    ReduceDocumentsChain,
    RetrievalQAWithSourcesChain,
    StuffDocumentsChain,
)
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models.openai import ChatOpenAI

from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser

# from langchain.parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.tools import BaseTool, Tool
from langchain.utilities import GoogleSearchAPIWrapper, GoogleSerperAPIWrapper
from langchain.vectorstores import Chroma
from llama_index import Prompt
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from pydantic import BaseModel, Field

# from langchain.llms import OpenAI
# from pathlib import Path


nest_asyncio.apply()

# For now I use my key
env = environ.Env()
environ.Env.read_env()
# OPENAI_API_KEY = env("OPENAI_API_KEY")
# GOOGLE_CSE_ID = env("GOOGLE_CSE_ID")
# GOOGLE_API_KEY = env("GOOGLE_API_KEY")
# SERPAPI_KEY = env("SERPAPI_KEY")
# SERPER_API_KEY = env('SERPER_API_KEY')
# openai.api_key = OPENAI_API_KEY


# load_dotenv(Path(".env"))
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
# openai.api_key = os.getenv('OPENAI_API_KEY')


# Vectorstore
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db_oai",
)

llm = ChatOpenAI(temperature=0)

# Initialize Custom Search with Google Programmable Search Engine
search = GoogleSearchAPIWrapper(google_api_key=OPENAI_API_KEY, google_cse_id=GOOGLE_CSE_ID)
# self.search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

# Initialize Web Research Retriever
web_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
    num_search_results=5,
)


# works but no sources provided
class QASourcesTool:
    def __init__(self):

        # Create the RetrievalQAWithSourcesChain
        self.qa_sources_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            retriever=web_retriever,
            chain_type="refine"
        )

    def run(self, user_input):
        return self.qa_sources_chain(
            {"question": user_input}, 
            # return_sources=True
            )

# OutputParserException: Could not parse LLM output
class QASourcesTool2():
    def __init__(self):

        self.reduce_template = """The following is a set of summaries:{doc_summaries}
            Take these and distill them into a final, consolidated summary of the main themes.
            Deliver the sources as Links in the answer.
            Answer:
            """
        
        self.reduce_prompt = PromptTemplate.from_template(self.reduce_template)

        self.reduce_chain = LLMChain(llm=llm, prompt=self.reduce_prompt)

        self.combine_documents_chain = StuffDocumentsChain(
            llm_chain=self.reduce_chain, 
            document_variable_name="doc_summaries"
        )

        self.qa_chain = RetrievalQAWithSourcesChain(
            # llm=llm, 
            retriever=web_retriever,
            combine_documents_chain=self.combine_documents_chain,
            # return_source_documents=True,
        )

    def run(self, query):
        result = self.qa_chain(query)
        # {"answer": result["answer"], "sources": result["sources"]}
        return result

# not working
# https://github.com/langchain-ai/langchain/issues/3523
class QASourcesTool3:
    def __init__(self):

        GERMAN_QA_PROMPT = PromptTemplate(
            template=f"", 
            input_variables=["question"],
            )
        
        GERMAN_DOC_PROMPT = PromptTemplate(
            template="Source: {source}\nContent: {page_content}",
            input_variables=["source", "content"])

        self.qa_chain = load_qa_with_sources_chain(
            llm, 
            chain_type="stuff",
            prompt=GERMAN_QA_PROMPT,
            document_prompt=GERMAN_DOC_PROMPT
            ) 
        
        self.chain = RetrievalQAWithSourcesChain(
            combine_documents_chain=self.qa_chain, 
            retriever=web_retriever,
            reduce_k_below_max_tokens=True, 
            max_tokens_limit=3375,
            return_source_documents=True
            )

    def run(self, user_input):
        return self.chain(
            {"question": user_input}, 
            )

# # works but no sources provided
class ChainToolFive():
    def __init__(self):
        # LLMChain
        self.search_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an assistant tasked with improving Google search 
            results. Generate FIVE Google search queries that are similar to
            this question. Transalte your queries into german as your searching for german official documents.
            The output should be a numbered list of questions and each
            should have a question mark at the end: {question}""",
        )

        class LineList(BaseModel):
            """List of questions."""

            lines: list[str] = Field(description="Questions")

        class QuestionListOutputParser(PydanticOutputParser):
            """Output parser for a list of numbered questions."""

            def __init__(self) -> None:
                super().__init__(pydantic_object=LineList)

            def parse(self, text: str) -> LineList:
                lines = re.findall(r"\d+\..*?\n", text)
                return LineList(lines=lines)
            
        self.llm_chain = LLMChain(
                    llm=llm,
                    prompt=self.search_prompt,
                    output_parser=QuestionListOutputParser(),
                )

        # retrieve docs and provide citations
        self.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, 
            retriever=web_retriever, 
            chain_type="refine",
        )

    def run(self, user_input):
        result = self.qa_chain({"question": user_input})
        return result

# works but no sources provided - refine
class RetrieverTool():
    def __init__(self):
        # Initialize LLM
        self.llm = llm

    def run(self, user_input):
        self.docs = web_retriever.get_relevant_documents(user_input)

        self.chain = load_qa_chain(
            self.llm, 
            chain_type="refine",
            return_refine_steps=True
            )

        return self.chain(
            {"input_documents": self.docs, "question": user_input},
            return_only_outputs=True, 
            )

# works talks about sources but does not provide them - map_reduce, combined custom prompt
class QAChain():
    def __init__(self):
        # Create the Prompt Template for base qa_chain
        self.qa_template = """Context information is below.
            ---------------------
            {context}
            ---------------------
            Given the context information and not prior knowledge, 
            answer the question: {question}
            You must deliver the relevant sources from {context} in your answer as links to websites.
            Answer:
        """

        self.PROMPT = PromptTemplate(
            template=self.qa_template, input_variables=["context", "question"]
        )

        # self.combine_prompt_template = """Given the following extracted parts of a long document and a question, 
        # create a final informative answer in english. 
        # You must deliver the relevant sources of the documents, especially links to the sources.
        # If you don't know the answer, just say that you don't know. Don't try to make up an answer.

        # QUESTION: {question}
        # =========
        # {summaries}
        # =========
        # Answer:
        # Sources:
        # """

        # self.COMBINE_PROMPT = PromptTemplate(
        #     template=self.combine_prompt_template, 
        #     input_variables=["summaries", "question"]
        # )

    def run(self, user_input):
        
        self.chain = load_qa_chain(
            llm, 
            chain_type="map_reduce", 
            return_map_steps=True, 
            question_prompt=self.PROMPT, 
            # combine_prompt=self.COMBINE_PROMPT
            )
        
        self.docs = web_retriever.get_relevant_documents(user_input)
        
        return self.chain(
            { "input_documents": self.docs, "question": user_input, },
            return_only_outputs=True,
            )

class QAChain2:
    def __init__(self):
        # Create the Prompt Template for base qa_chain
        self.qa_template = """Context information is below.
            ---------------------
            {context}
            ---------------------
            Given the context information and not prior knowledge, 
            answer the question: {question}
            Deliver the relevant sources from {context} in your answer as links to websites.
            Answer:
        """

        self.PROMPT = PromptTemplate(
            template=self.qa_template, input_variables=["context", "question"]
        )

        self.combine_prompt_template = """Given the following extracted parts of a long document and a question, 
        create a final informative answer in english. 
        You must deliver the relevant sources of the documents, especially links to the sources.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.

        QUESTION: {question}
        =========
        {summaries}
        =========
        Document Sources:
        {sources}

        Answer:
        Sources:"""

        self.COMBINE_PROMPT = PromptTemplate(
            template=self.combine_prompt_template, input_variables=["summaries", "question", "sources"]
        )

    def run(self, user_input):
        # Retrieve documents and sources from web retriever
        documents = web_retriever.get_relevant_documents(user_input)
        
        sources = []
        for doc in documents:
            sources.append(doc.metadata['sources'])

        # Combine document sources into the required format
        sources_str = '\n'.join(sources)

        # Create the QA chain
        self.chain = load_qa_chain(
            llm, 
            chain_type="map_reduce", 
            return_map_steps=True, 
            question_prompt=self.PROMPT, 
            combine_prompt=self.COMBINE_PROMPT
        )
        
        # Call the QA chain with the constructed prompts
        return self.chain(
            {
                "input_documents": documents,
                "question": user_input,
                "sources": sources_str,
            },
            return_only_outputs=True,
        )

class EngineTool():
    def __init__(self):
        self.qa_template = Prompt(
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Do not give me an answer if it is not mentioned in the context as a fact. \n"
            "Given this information, please provide me with an answer to the following question:\n{query_str}\n"
        )


        # set up query engine from embedding
        self.number_top_results = 3  
        self.folder_with_index = "vector_db"
        self.query_engine_default = default_engine(self.folder_with_index, self.qa_template, self.number_top_results )

        # define query engine as a tool for langchain agent
        self.tool_config = IndexToolConfig(
            query_engine=self.query_engine_default, 
            name=f"Vector Index",
            description=f"useful for when you want to answer queries about work and unemployment in Germany",
            tool_kwargs={"return_direct": False}
        )

    def run(self):
        return LlamaIndexTool.from_tool_config(self.tool_config)

class ToolChainAgent():
    def __init__(self):
        self.llm = llm
        self.web_retriever = web_retriever

        # initialize tools 
        self.q_engine_fct = EngineTool().run()
        self.retriever_fct = RetrieverTool().run # works but no sources provided
        # self.retriever_fct = QASourcesTool().run # works but no sources provided
        # self.retriever_fct = ChainToolFive().run # works but no sources provided
        # self.retriever_fct = QAChain().run # works talks about sources but does not deliver them
        # self.retriever_fct = QAChain2().run # 
        # self.retriever_fct = QASourcesTool2().run




        # # Extra tools
        # self.gsearch_func = lambda query: GoogleSearchAPIWrapper()(query)
        # self.serper_func = lambda query: GoogleSerperAPIWrapper()(query)


        self.tools = [
            # embedding first
            Tool( name="query engine", 
                description="use this tool first, to get anwsers about work and unemployment in berlin", 
                func=self.q_engine_fct),

            # web retriever goes through programmable search engine
            Tool( name="web_retriever", 
                description="use this tool when the first failed", 
                func=self.retriever_fct),

            # # usual web search
            # Tool( name="google_search", 
            #     description="use this tool if the first tools does not give you a sytifying answer", 
            #     func=self.gsearch_func), 

            # # usual web search
            # Tool( name="serper", 
            #     description="use this tool if the other tools do not give you an satisfying answer", 
            #     func=self.serper_func), 
        ]

        ## to add memory to the agent
        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,

        self.agent = initialize_agent(
            self.tools,
            ChatOpenAI(temperature=0),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            # memory=memory,
        )

    def run(self, user_input):
        """Agent runs query with the listed tools: 

        1. query engine from custom embedding
        2. web retriever, which goes through programmable google search

        - run the agent instance like this: agent.run(promt)"""
        return self.agent(user_input)
