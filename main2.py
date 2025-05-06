from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from crew import run_crew  
import io
from loguru import logger
from typing import Type

if __name__ == "__main__":
    logger.debug("Start")

    # Static inputs
    query = "Give me happiest countries of the least happy continent."
    file_path = "2017.csv"

    # Validate query
    logger.debug(query)
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Check file type
    logger.debug(file_path)
    if not file_path.endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    # Read CSV content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        data = pd.read_csv(io.StringIO(content))
        logger.success(data)
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")

    # Validate data
    if data.empty:
        raise ValueError("Uploaded CSV is empty.")

    # Run CrewAI agent pipeline
    try:
        stringified_data = data.to_csv()
        result = run_crew(query, data)
        logger.critical(type(result))
    except Exception as e:
        raise RuntimeError(f"CrewAI execution error: {str(e)}")

    print({"status": "success", "output": result})