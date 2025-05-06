from crewai import Agent
from typing import Type

from dotenv import load_dotenv

load_dotenv()

import os
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"

from crewai import Agent
from tools import HandleMissingDataTool, HandleOutliersTool, ScaleDataTool, EngineerFeaturesTool, ParseDateColumnsTool, HandleCategoricalDataTool, SelectFeaturesTool, ConvertDataTypesTool

handle_missing = HandleMissingDataTool()
handle_outliers = HandleOutliersTool()
scale_data = ScaleDataTool()
engineer_features = EngineerFeaturesTool()
parse_dates = ParseDateColumnsTool()
handle_categorical = HandleCategoricalDataTool()
select_features = SelectFeaturesTool()
convert_types = ConvertDataTypesTool()

data_analyst_agent = Agent(
    name="DataAnalyst",
    role="Data Preprocessing Specialist",
    goal="Clean, preprocess, and engineer features from raw data based on the user query",
    backstory=(
        "A meticulous data analyst who understands the connection between raw data and business queries. "
        "Expert at cleaning, formatting, and enriching data before analysis."
    ),
    tools=[
        handle_missing,
        handle_outliers,
        scale_data,
        engineer_features,
        parse_dates,
        handle_categorical,
        select_features,
        convert_types
    ],
    verbose=True,
)



