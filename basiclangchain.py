from openai import OpenAI
import pandas as pd
from loguru import logger
import json


df = pd.read_csv("2017.csv")

user_query = "Show me relationship between happiness index and generosity index."
logger.info(user_query)

def get_column_metadata(df: pd.DataFrame, sample_size: int = 5):
    metadata = {}
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_date_like = pd.to_datetime(series, errors='coerce').notna().mean() > 0.7

        metadata[col] = {
            "dtype": str(series.dtype),
            "is_numeric": is_numeric,
            "is_date_like": is_date_like,
            "n_missing": int(series.isnull().sum()),
            "n_unique": int(series.nunique()),
            "sample_values": series.dropna().unique().tolist()[:sample_size]
        }

        if is_numeric:
            metadata[col].update({
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std()
            })

    return metadata

column_metadata = get_column_metadata(df)

#logger.info(column_metadata)




DATA_PREPROCESSING_AGENT_PROMPT = f"""
You are a data preprocessing expert.

Given the following user query, columns and column metadata decide which preprocessing steps should be applied and which columns each step should target from the given columns under "## Columns".

## Available Preprocessing Functions:
1. handle_missing
2. handle_outliers
3. scale_data
4. engineer_features
5. parse_dates
6. handle_categorical
7. select_features
8. convert_types

## Guidelines:
- You can skip functions that are not needed.
- A column can be included in multiple preprocessing steps.
- Return output in this exact nested list format:
  [
    [<list of preprocessing function names used>],
    [<list of lists of corresponding column names, in the same order as above>]
  ]

## Example:
[
    ["handle_missing", "handle_outliers", "handle_categorical"],
    [["Insulin", "BMI"], ["Age", "Pregnancies"], ["Gender"]]
]

Only return the list. Do not include any explanation or extra content.

## User Query:
{user_query}

## Columns:
{df.columns.tolist()}

## Column Metadata:
{column_metadata}
"""

logger.critical(DATA_PREPROCESSING_AGENT_PROMPT)

# Call OpenAI LLM (replace with your actual OpenAI API call logic)
client = OpenAI(}

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # or gpt-3.5-turbo if preferred
    messages=[
        {"role": "system", "content": "You are an expert data analyst."},
        {"role": "user", "content": DATA_PREPROCESSING_AGENT_PROMPT}
    ],
    temperature=0.2
)

# Extract structured list
decision_output = response.choices[0].message.content
logger.critical(type(decision_output))
logger.success(decision_output)
