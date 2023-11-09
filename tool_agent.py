import logging
import os

from bot_utils import default_engine
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import Chroma
from llama_index import Prompt
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool

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
        qa_template = Prompt(
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Do not give me an answer if it is not mentioned in the context as a fact. \n"
            "Given this information, please provide me with an answer to the following question:\n{query_str}\n"
        )

        self.folder_with_index = "vector_db"
        self.qa_template = qa_template
        self.number_top_results = 5

        self.default_engine = default_engine( self.folder_with_index, self.qa_template, self.number_top_results)

        # ERROR: IndexToolConfig awaits an instance of BaseQueryEngine
        self.tool_config = IndexToolConfig(
            query_engine=default_engine,
            name=f"Vector Index",
            description=f"Useful for when you want to answer queries about work and unemployment in Berlin, Germany",
            tool_kwargs={"return_direct": True},
        )

    def run(self):
        return LlamaIndexTool.from_tool_config(self.tool_config)


class RetrieverTool:
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
