from crewai import Crew,Process
from agents import data_analyst_agent
from tasks import data_cleaning_task
import pandas as pd
import os
from groq import Groq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Type
from loguru import logger
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

load_dotenv()  # This loads variables from .env into the environment

groq_api_key = os.getenv("GROQ_API_KEY")

def run_crew(query: str, data: str):

    query_s = StringKnowledgeSource(
    content=query,
)
    data_s = StringKnowledgeSource(
    content=data,
)
    logger.debug(data_s)
    '''
    data_s = CSVKnowledgeSource(
    file_paths= ['C:\\Users\\dell\\Desktop\\CrewConda\\2017.csv']
)
    '''
    data_analyst_agent.knowledge_sources = [data_s]
    data_cleaning_task.output = {
    "query": query,
    "data": data
}
    logger.critical(data_cleaning_task)
    gapi = os.getenv("GROQ_API_KEY")
    logger.info(gapi)


    llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=gapi,
        base_url="https://api.groq.com/openai/v1"
    )

    logger.debug(llm)
    crew = Crew(
        agents=[data_analyst_agent],
        tasks=[data_cleaning_task],
        llm=llm,
        verbose=True
    )

    logger.debug(crew)
    # Pass the inputs (query and DataFrame) to the Crew
    logger.info("Starting Crew kickoff")
    try:
        results = crew.kickoff(inputs={
        "query": query,
        "data": data
    })
        logger.critical(f"Results: {results}")
    except Exception as e:
        logger.exception(f"Error during kickoff: {e}")
        raise e

    return str(results)



