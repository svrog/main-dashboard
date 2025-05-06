import pandas as pd
import json
from typing import Optional, TypedDict
from langgraph.graph import StateGraph
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from loguru import logger

# ----------------------------
# Define the shared graph state
# ----------------------------
class GraphState(TypedDict):
    raw_csv: str
    query: str
    df: Optional[pd.DataFrame]
    column_metadata: Optional[dict]
    preprocessing_plan: Optional[list]

# ----------------------------
# Node 1: Load CSV into DataFrame
# ----------------------------
def load_csv(state: GraphState) -> GraphState:
    df = pd.read_csv(state["raw_csv"])
    return {**state, "df": df}

# ----------------------------
# Node 2: Analyze Column Metadata
# ----------------------------
def get_column_metadata(df: pd.DataFrame, sample_size: int = 5) -> dict:
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

def analyze_metadata(state: GraphState) -> GraphState:
    df = state["df"]
    metadata = get_column_metadata(df)
    return {**state, "column_metadata": metadata}

# ----------------------------
# Node 3: Decide Preprocessing Steps using LLM
# ----------------------------
AVAILABLE_PREPROCESSING_FUNCTIONS = [
    "handle_missing",
    "handle_outliers",
    "scale_data",
    "engineer_features",
    "parse_dates",
    "handle_categorical",
    "select_features",
    "convert_types"
]

client = OpenAI() # Replace with your key


def generate_preprocessing_prompt(column_metadata: dict) -> str:
    logger.critical(type(column_metadata))
    try:
        # Convert column metadata to JSON format
        column_metadata_json = json.dumps(column_metadata, indent=2)
        logger.debug(f"Column metadata JSON: {column_metadata_json}")
    except TypeError as e:
        print(f"Error serializing column_metadata: {e}")
        column_metadata_json = str(column_metadata)
        logger.debug(f"Column metadata JSON: {column_metadata_json}")
    return f"""
You are a data preprocessing expert.

Given the following column metadata, decide which preprocessing steps should be applied and which columns each step should target.

## Available Preprocessing Functions:
{', '.join(AVAILABLE_PREPROCESSING_FUNCTIONS)}

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

## Column Metadata:
{json.dumps(column_metadata_json, indent=2)}
"""

def decide_preprocessing_steps(state: GraphState) -> GraphState:
    prompt = generate_preprocessing_prompt(state["column_metadata"])
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    decision_output = response.choices[0].message.content.strip()

    try:
        preprocessing_plan = json.loads(decision_output)
    except Exception as e:
        raise ValueError(f"Failed to parse LLM output: {decision_output}") from e

    return {**state, "preprocessing_plan": preprocessing_plan}

# ----------------------------
# Preprocessing Functions
# ----------------------------
def handle_missing(df, columns):
    return df.fillna(df.mean(numeric_only=True)) if columns else df

def handle_outliers(df, columns):
    # Simple z-score-based outlier removal
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            z_score = (df[col] - df[col].mean()) / df[col].std()
            df = df[(z_score < 3) & (z_score > -3)]
    return df

def scale_data(df, columns):
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-9)
    return df

def engineer_features(df, columns):
    return df  # Placeholder

def parse_dates(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def handle_categorical(df, columns):
    return pd.get_dummies(df, columns=columns) if columns else df

def select_features(df, columns):
    return df[columns] if columns else df

def convert_types(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

preprocessing_function_map = {
    "handle_missing": handle_missing,
    "handle_outliers": handle_outliers,
    "scale_data": scale_data,
    "engineer_features": engineer_features,
    "parse_dates": parse_dates,
    "handle_categorical": handle_categorical,
    "select_features": select_features,
    "convert_types": convert_types
}

def apply_preprocessing(state: GraphState) -> GraphState:
    df = state["df"].copy()
    plan = state["preprocessing_plan"]
    if not plan:
        return state

    function_names, column_lists = plan
    for func_name, cols in zip(function_names, column_lists):
        func = preprocessing_function_map.get(func_name)
        if func:
            df = func(df, cols)

    return {**state, "df": df}

VISUALIZATION_DECISION_PROMPT = """
You are a data visualization expert.

Given the metadata of a preprocessed DataFrame, suggest a list of visualizations that best represent the data.

For each visualization, return:
- type: one of ["bar", "line", "scatter", "histogram", "box", "pie"]
- columns: list of column names used
- title: short descriptive title

Only return a list of dictionaries in this format:
[
  {{ "type": "bar", "columns": ["Category", "Sales"], "title": "Sales by Category" }},
  ...
]
Do NOT include explanations.

## Metadata:
{column_metadata}
"""
def decide_visualizations(state: GraphState) -> GraphState:
    df = state["df"]
    column_metadata = get_column_metadata(df)

    prompt = VISUALIZATION_DECISION_PROMPT.format(column_metadata=json.dumps(column_metadata, indent=2))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    decision_output = response.choices[0].message.content
    try:
        viz_plan = json.loads(decision_output)
    except Exception as e:
        print("Failed to parse visualization decision:", e)
        viz_plan = []
    return {**state, "visualization_plan": viz_plan}

def generate_visualizations(state: GraphState) -> GraphState:
    df = state["df"]
    viz_plan = state["visualization_plan"]
    
    generated_viz = []
    
    for viz in viz_plan:
        viz_type = viz.get("type")
        columns = viz.get("columns")
        title = viz.get("title")
        
        if viz_type == "bar":
            generated_viz.append(generate_bar_chart(df, columns, title))
        elif viz_type == "line":
            generated_viz.append(generate_line_chart(df, columns, title))
        elif viz_type == "scatter":
            generated_viz.append(generate_scatter_plot(df, columns, title))
        elif viz_type == "histogram":
            generated_viz.append(generate_histogram(df, columns, title))
        elif viz_type == "box":
            generated_viz.append(generate_box_plot(df, columns, title))
        elif viz_type == "pie":
            generated_viz.append(generate_pie_chart(df, columns, title))
    
    return {**state, "generated_viz": generated_viz}

# Helper functions to generate each chart type

def generate_bar_chart(df, columns, title):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=columns[0], y=columns[1], data=df)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"bar_chart_{title}.png")
    return f"bar_chart_{title}.png"

def generate_line_chart(df, columns, title):
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=columns[0], y=columns[1], data=df)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"line_chart_{title}.png")
    return f"line_chart_{title}.png"

def generate_scatter_plot(df, columns, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=columns[0], y=columns[1], data=df)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"scatter_plot_{title}.png")
    return f"scatter_plot_{title}.png"

def generate_histogram(df, columns, title):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[columns[0]], kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"histogram_{title}.png")
    return f"histogram_{title}.png"

def generate_box_plot(df, columns, title):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=columns[0], y=columns[1], data=df)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"box_plot_{title}.png")
    return f"box_plot_{title}.png"

def generate_pie_chart(df, columns, title):
    plt.figure(figsize=(8, 6))
    df[columns[0]].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"pie_chart_{title}.png")
    return f"pie_chart_{title}.png"


def create_dashboard(state: GraphState) -> GraphState:
    # Get the list of generated visualizations (image paths)
    generated_viz = state["generated_viz"]
    
    # Create the dashboard using Streamlit
    st.title("Automated Data Visualization Dashboard")
    
    # Display each visualization in the list
    for viz in generated_viz:
        image_path = os.path.join(os.getcwd(), viz)  # Assuming images are in the current working directory
        if os.path.exists(image_path):
            st.image(image_path, caption=viz, use_column_width=True)
        else:
            st.error(f"Image {viz} not found!")
    
    # Return the state with dashboard created (this step is optional depending on the use case)
    return state




# ----------------------------
# Build LangGraph
# ----------------------------
builder = StateGraph(GraphState)

builder.add_node("load_csv", load_csv)
builder.add_node("analyze_metadata", analyze_metadata)
builder.add_node("decide_preprocessing", decide_preprocessing_steps)
builder.add_node("apply_preprocessing", apply_preprocessing)
builder.add_node("decide_visualizations", decide_visualizations)
builder.add_node("generate_visualizations", generate_visualizations)
builder.add_node("create_dashboard", create_dashboard)

builder.set_entry_point("load_csv")
builder.add_edge("load_csv", "analyze_metadata")
builder.add_edge("analyze_metadata", "decide_preprocessing")
builder.add_edge("decide_preprocessing", "apply_preprocessing")
builder.add_edge("apply_preprocessing", "decide_visualizations")
builder.add_edge("decide_visualizations", "generate_visualizations")
builder.add_edge("generate_visualizations", "create_dashboard")
builder.set_finish_point("create_dashboard")

graph = builder.compile()

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    input_state = {
        "raw_csv": "2017.csv",  # <-- Replace with your CSV path
        "query": "Create a visualization dashboard for data insights.",
        "df": None,
        "column_metadata": None,
        "preprocessing_plan": None
    }

    final_state = graph.invoke(input_state)
    print(json.dumps(final_state["preprocessing_plan"], indent=2))
