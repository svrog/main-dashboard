from crewai import Task
from agents import data_analyst_agent
from loguru import logger


logger.info(Task)



data_cleaning_task = Task(
    description=(
        "Analyze the user query {query} and the input data (a Pandas DataFrame). "
        "Use the LLM to decide which data preprocessing steps are necessary, "
        "such as handling missing values, outliers, feature scaling, feature engineering, etc. "
        "Only apply those tools that are relevant to the data and query."
    ),
    expected_output=(
        "A cleaned and preprocessed DataFrame that is ready for analysis and visualization."
    ),
    agent=data_analyst_agent,
    async_execution=False,
    output_type="pandas.DataFrame",
    
)

