from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from crew import run_crew  
import io
from loguru import logger
from typing import Type

app = FastAPI()

@app.post("/create-dashboard/")
async def analyze_data(query: str = Form(...), file: UploadFile = File(...)):
    logger.debug("Start")
    try:
        # Validate query
        logger.debug(query)
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # Check file type
        logger.debug(file.filename)
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        # Read CSV content
        content = await file.read()
        #logger.debug(content)
        try:
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
            logger.success(data)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

        # Validate data
        if data.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
            

        # Run CrewAI agent pipeline
        try:
            stringified_data = data.to_csv()
            result = run_crew(query, stringified_data)
            logger.critical(type(result))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"CrewAI execution error: {str(e)}")

        return JSONResponse(content={"status": "success", "output": result})

    except HTTPException as http_err:
        raise http_err  # re-raise known errors

    except Exception as err:
        # Unexpected errors
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {str(err)}"}
        )
