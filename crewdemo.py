import pandas as pd
from crewai import Agent, Crew, Task
from langchain_community.chat_models import ChatOpenAI
import os 
import io
from dotenv import load_dotenv
from io import StringIO
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from groq import Groq
from crewai import LLM


gapi = os.getenv("GROQ_API_KEY")


class PreprocessInput(BaseModel):
    """Input schema for all preprocessing tools."""
    query: str = Field(..., description="User query or instruction for preprocessing.")
    dataframe: str = Field(..., description="Stringified DataFrame to process.")


class DropNullsTool(BaseTool):
    name: str = "Drop Null Values"
    description: str = "Drops all null values from the dataset."
    args_schema: Type[BaseModel] = PreprocessInput

    def _run(self, query: str, dataframe: str) -> str:
        df = pd.read_csv(io.StringIO(dataframe))
        df.dropna(inplace=True)
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

class NormalizeNumericTool(BaseTool):
    name: str = "Normalize Numeric Columns"
    description: str = "Normalizes numeric columns in the dataset using Min-Max scaling."
    args_schema: Type[BaseModel] = PreprocessInput

    def _run(self, query: str, dataframe: str) -> str:
        df = pd.read_csv(io.StringIO(dataframe))
        numeric = df.select_dtypes(include='number')
        df[numeric.columns] = (numeric - numeric.min()) / (numeric.max() - numeric.min())
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

class EncodeCategoricalTool(BaseTool):
    name: str = "Encode Categorical Columns"
    description: str = "Encodes categorical (object) columns as integers."
    args_schema: Type[BaseModel] = PreprocessInput

    def _run(self, query: str, dataframe: str) -> str:
        df = pd.read_csv(io.StringIO(dataframe))
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category').cat.codes
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

drop_nulls = DropNullsTool()
normalize_numeric = NormalizeNumericTool()
encode_categorical = EncodeCategoricalTool()

preprocess_agent = Agent(
    role="Data Preprocessing Expert",
    goal="Analyze the dataset and apply the best preprocessing step",
    backstory="You are a data preprocessing expert. Based on data inspection, you choose the best tool to clean or transform the data.",
    tools=[drop_nulls, normalize_numeric, encode_categorical],
    verbose=True
)










#Client = Groq(api_key=gapi, base_url="https://api.groq.com/openai/v1")
'''
llm = LLM(
    model="llama-3.1-8b-instant",  # Or "llama3-8b-8192"
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",  # Groq's OpenAI-compatible endpoint
)
'''
oapi = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
        model="gpt-4o mini",
        temperature=0.3,
        api_key=oapi,
    )





# Load and summarize dataset
df = pd.read_csv("2017.csv")

# Extract structured insights
info_buf = StringIO()
df.info(buf=info_buf)
info_text = info_buf.getvalue()

head_text = df.head().to_string()
desc_text = df.describe(include="all").to_string()

dataset_summary = f"""
### Dataset Summary:

[INFO]
{info_text}

[HEAD]
{head_text}

[DESCRIBE]
{desc_text}
"""

# Task with embedded summary
preprocess_task = Task(
    description=(
        "You are provided with a dataset summary {dataset_summary}. Based on this, determine which preprocessing steps are needed. "
        "Choose from: drop_nulls, normalize_numeric, encode_categorical. "
        "Apply only what's required. Then save the cleaned data to 'data/preprocessed_data.csv'.\n\n"
        f"{dataset_summary}"
    ),
    expected_output="Preprocessed data saved successfully using the selected tool(s).",
    output_file="2017output.csv",
    agent=preprocess_agent
)

# Run the crew
#llm = client 

crew = Crew(
    agents=[preprocess_agent],
    tasks=[preprocess_task],
    llm=llm,
    verbose=True
)

if __name__ == "__main__":
    results = crew.kickoff()
    strres = str(results)
    df = pd.read_csv(StringIO(strres))

    df.to_csv("2017output.csv", index=False)