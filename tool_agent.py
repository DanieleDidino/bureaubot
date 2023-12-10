import logging
import os
import re
from typing import List

from bot_utils import default_engine
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, tool
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.tools import BaseTool, Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import Chroma
from llama_index import Prompt, StorageContext, load_index_from_storage
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from pydantic import BaseModel, Field

# logging to see what sites and documents the web retriever is using
logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)

# Search
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

search = GoogleSearchAPIWrapper()
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


class EngineTool:
    def __init__(self):
        
        self.qa_template = Prompt(
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Do not give me an answer if it is not mentioned in the context as a fact. \n"
            "Given this information, please provide me with an answer to the following question:\n{query_str}\n"
        )

        self.folder_with_index = "vector_db"
        self.number_top_results = 5

        self.default_engine = default_engine(
            self.folder_with_index, self.qa_template, self.number_top_results
        )

        # ERROR: IndexToolConfig awaits an instance of BaseQueryEngine
        self.tool_config = IndexToolConfig(
            query_engine=default_engine,
            name=f"Vector Index",
            description=f"Useful for when you want to answer queries about work and unemployment in Berlin, Germany",
            tool_kwargs={"return_direct": True},
        )

    def run(self):
        return LlamaIndexTool.from_tool_config(self.tool_config)

class IndexTool2:
    def __init__(self):
        self.qa_template = PromptTemplate(
            """
            You are a helpful administrative assistant that help the user with legalese and bureaucracy.
            Only give answers based on facts which sources you can provide.

            We have provided context information below. \n
            ---------------------\n
            {context_str}
            ---------------------\n
            
            Given this information, please answer the question: {query_str}\n
            """)

        ## Index tool from documents in vector store
        self.storage_context = StorageContext.from_defaults(persist_dir="vector_db")
        self.index = load_index_from_storage(self.storage_context)

        self.query_engine = self.index.as_query_engine(
            streaming=False, text_qa_template=self.qa_template, similarity_top_k=5
        )
        
        self.tool_config = IndexToolConfig(
            query_engine=self.query_engine,
            name=f"Vector Index",
            description=f"useful for when you want to answer queries about work and unemployment in Berlin, Germany",
            tool_kwargs={"return_direct": True},
        )

    def run(self):
        return LlamaIndexTool.from_tool_config(self.tool_config)

class RetrieverTool:
    """Tool to find information about work, unemployment, laws and adminstrative infromation in Berlin, Germany"""

    def __init__(self):
        # Initialize LLM
        self.llm = llm

        # Vectorstore
        self.vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
        )

        # Initialize
        self.web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=self.vectorstore,
            llm=self.llm,
            search=search,
        )

    def run(self, user_input):
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            self.llm, retriever=self.web_research_retriever
        )

        return qa_chain({"question": user_input})

class RetrieverTool2:
    """Tool to find information about work, unemployment, laws and adminstrative infromation in Berlin, Germany"""
    
    def __init__(self):
        self.llm = llm
    
        self.search_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an assistant tasked with improving Google search 
            results. Generate FIVE Google search queries that are similar to
            this question. The output should be a numbered list of questions and each
            should have a question mark at the end: {question}"""
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


        self.llm_chain = LLMChain(
            llm=llm,
            prompt=self.search_prompt,
            output_parser=QuestionListOutputParser(),
        )


        # Vectorstore
        self.vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
        )

        # Search
        self.search = GoogleSearchAPIWrapper()

        # Initialize
        self.web_research_retriever_llm_chain = WebResearchRetriever(
            vectorstore=self.vectorstore,
            llm_chain=self.llm_chain,
            search=self.search,
        )

        
    def run(user_input):
        docs = self.web_research_retriever_llm_chain.get_relevant_documents(user_input)
        chain = load_qa_chain(llm, chain_type="stuff")
        return chain( {"input_documents": docs, "question": user_input}, return_only_outputs=True )

class ToolChainAgent:
    def __init__(self):
        self.llm = llm

        # initialize tools
        self.q_engine_fct = EngineTool().run()
        self.retriever_fct = RetrieverTool(self.llm).run

        self.tools = [
            # embedding first
            Tool(
                name="query engine",
                description="use this tool first, to get anwsers about work and unemployment in Berlin, Germany",
                func=self.q_engine_fct,
            ),
            # web retriever goes through programmable search engine
            Tool(
                name="web_retriever",
                description="use this tool when the first tool failed and you want to answer questions about work and unempoyment in Berlin, Germany",
                func=self.retriever_fct,
            ),
        ]

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def run(self, user_input):
        """
        Agent runs query with the listed tools:

        1. query engine from custom embedding
        2. web retriever, which goes through programmable google-search

        Run the agent instance like this:
        agent = ToolChainAgent()
        agent.run(promt)
        """
        return self.agent(user_input)

class ToolChainAgent2:
    def __init__(self):
        self.llm = llm

        # initialize tools
        self.q_engine_fct = IndexTool2().run()
        self.retriever_fct = RetrieverTool2().run

        self.tools = [
            # embedding first
            Tool(
                name="query engine",
                description="Use to get anwsers about work and unemployment in Berlin, Germany",
                func=lambda q: str(self.q_engine_fct.query(q)),
            ),
            # web retriever goes through programmable search engine
            Tool(
                name="web_retriever",
                description="Use this tool when you want to answer questions about administration, laws and bureaucracy in Berlin, Germany",
                func=self.retriever_fct,
            ),
        ]

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def run(self, user_input):
        """
        Agent runs query with the listed tools:

        1. query engine from custom embedding
        2. web retriever, which goes through programmable google-search

        Run the agent instance like this:
        agent = ToolChainAgent()
        agent.run(promt)
        """
        return self.agent(user_input)