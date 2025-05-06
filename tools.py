import numpy as np
import pandas as pd
import io
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from typing import Type

class PreprocessInput(BaseModel):
    """Input schema for all preprocessing tools."""
    query: str = Field(..., description="User query or instruction for preprocessing.")
    dataframe: str = Field(..., description="Stringified DataFrame to process.")

class HandleMissingDataTool(BaseTool):
    name: str = "Handle Missing Data"
    description: str = "Handles missing values in the dataset using LLM context."
    args_schema: Type[BaseModel] = PreprocessInput


    def handle_missing_data(data, method='drop', fill_value=None):
        """Handle missing data in a DataFrame."""
        if method == 'drop':
            return data.dropna()
        elif method == 'fill':
            return data.fillna(fill_value)
        else:
            print("Invalid method. Use 'drop' or 'fill'.")
            return data
        
    def _run(self,query: str, dataframe: str) -> str:
        data = pd.read_csv(io.StringIO(dataframe))(da)
        return self.handle_missing_data(data)
    
class HandleOutliersTool(BaseTool):
    name: str = "Handle Outliers"
    description: str = "Detects and handles outliers in the dataset."
    args_schema: Type[BaseModel] = PreprocessInput

    
    def handle_outliers(data, threshold=3):
        """Detect and remove outliers based on Z-score."""
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[(z_scores < threshold).all(axis=1)]
    
    def _run(self,query, dataframe: str) -> str:
        data = pd.read_csv(io.StringIO(dataframe))
        return self.handle_outliers(data)

class ScaleDataTool(BaseTool):
    name: str = "Scale Data"
    description: str = "Scales numerical features based on query and context."
    args_schema: Type[BaseModel] = PreprocessInput

    def scale_data(data, method='standard'):
        """Scale or normalize data using specified method."""
        if method == 'standard':
            scaler = StandardScaler()
            return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        else:
            print("Invalid method. Use 'standard' or 'minmax'.")
            return data
        
    def _run(self,query, dataframe: str) -> str:
        data = pd.read_csv(io.StringIO(dataframe))
        return self.scale_data(data)

class EngineerFeaturesTool(BaseTool):
    name: str = "Engineer Features"
    description: str = "Creates new features based on query context."
    args_schema: Type[BaseModel] = PreprocessInput

    def engineer_features(data):
        """Generate new features or handle categorical columns."""
        # Example: One-hot encode categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_data = encoder.fit_transform(data[categorical_cols])
        
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(categorical_cols))
        data = pd.concat([data, encoded_df], axis=1).drop(categorical_cols, axis=1)
        return data
    
    def _run(self,query, dataframe: str) -> str:
        data = pd.read_csv(io.StringIO(dataframe))
        return self.engineer_features(data)

class ParseDateColumnsTool(BaseTool):
    name: str = "Parse Date Columns"
    description: str = "Parses and enhances date columns in the data."
    args_schema: Type[BaseModel] = PreprocessInput

    def parse_date_columns(data):
        """Convert date-time columns to year, month, day, etc."""
        date_columns = data.select_dtypes(include=['object']).columns
        for col in date_columns:
            try:
                data[col] = pd.to_datetime(data[col])
                data[f'{col}_year'] = data[col].dt.year
                data[f'{col}_month'] = data[col].dt.month
                data[f'{col}_day'] = data[col].dt.day
                data[f'{col}_hour'] = data[col].dt.hour
            except Exception as e:
                print(f"Error processing date column {col}: {e}")
        return data

    def _run(self,query, dataframe: str) -> str:
        data = pd.read_csv(io.StringIO(dataframe))
        return self.parse_date_columns(data)
    
class HandleCategoricalDataTool(BaseTool):
    name: str = "Handle Categorical Data"
    description: str = "Encodes categorical variables as per query intent."
    args_schema: Type[BaseModel] = PreprocessInput

    def handle_categorical_data(data):
        """One-hot encode or label encode categorical columns."""
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        return data
    
    def _run(self,query, dataframe: str) -> str:
        data = pd.read_csv(io.StringIO(dataframe))
        return self.handle_categorical_data(data)
    
class SelectFeaturesTool(BaseTool):
    name: str = "Select Relevant Features"
    description: str = "Selects features relevant to the user's query."
    args_schema: Type[BaseModel] = PreprocessInput

    def select_features(data, target, k=10):
        """Select top k features based on statistical tests."""
        X = data.drop(target, axis=1)
        y = data[target]
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        return data[selected_features]
    
    def _run(self,query, dataframe: str) -> str:
        data = pd.read_csv(io.StringIO(dataframe))
        return self.select_features(data)

class ConvertDataTypesTool(BaseTool):
    name: str = "Convert Data Types"
    description: str = "Casts data to correct types as per context and query."
    args_schema: Type[BaseModel] = PreprocessInput

    def convert_data_types(data):
        """Convert columns to appropriate data types."""
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_datetime(data[col], errors='ignore')
                except:
                    data[col] = data[col].astype('category')
            elif data[col].dtype == 'int64' or data[col].dtype == 'float64':
                data[col] = pd.to_numeric(data[col], errors='coerce')
        return data
    
    def _run(self,query, dataframe: str) -> str:
        data = pd.read_csv(io.StringIO(dataframe))
        return self.convert_data_types(data)







