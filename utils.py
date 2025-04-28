import io
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
#import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_data(uploaded_file) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load data from uploaded file into a pandas DataFrame."""
    try:
        # Read the file based on type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            return None, f"Unsupported file format: {uploaded_file.name}"
        
        return df, None
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

@st.cache_resource
def load_huggingface_model():
    """Load a Hugging Face model for text generation. Cached by Streamlit."""
    try:
        # Using a smaller model optimized for text generation
        model_name = "facebook/opt-350m"  # A relatively small model
        
        # For text generation
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def analyze_with_huggingface(df: pd.DataFrame, query: str, tokenizer, model) -> str:
    """Use Hugging Face model to analyze the dataframe based on the query."""
    if tokenizer is None or model is None:
        return "Model failed to load. Using rule-based analysis only."
    
    # Create a context about the dataframe
    df_info = f"""
    DataFrame Information:
    - Shape: {df.shape}
    - Columns: {', '.join(df.columns.tolist())}
    - Data types: {df.dtypes.to_string()}
    - First few rows: 
    {df.head(3).to_string()}
    """
    
    # Create the prompt
    prompt = f"""
    I have a pandas DataFrame with the following information:
    {df_info}
    
    User question: {query}
    
    Data Analysis:
    """
    
    try:
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(
            inputs.input_ids, 
            max_length=750,
            num_return_sequences=1,
            temperature=0.7
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (after "Data Analysis:")
        if "Data Analysis:" in response:
            response = response.split("Data Analysis:")[1].strip()
        
        return response
    except Exception as e:
        return f"Error with Hugging Face model: {str(e)}"

def perform_rule_based_analysis(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """Perform rule-based analysis based on keywords in query."""
    results = {}
    
    # Basic summary statistics
    if any(word in query.lower() for word in ["summary", "describe", "statistics", "stats"]):
        results["summary"] = df.describe()
    
    # Correlation analysis
    if any(word in query.lower() for word in ["correlation", "correlate", "relationship"]):
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            results["correlation"] = numeric_df.corr()
    
    # Missing values analysis
    if any(word in query.lower() for word in ["missing", "null", "na", "nan"]):
        missing = df.isnull().sum()
        if missing.sum() > 0:
            results["missing"] = missing[missing > 0]
        else:
            results["missing"] = "No missing values found in the dataset!"
    
    # Unique value counts
    if any(word in query.lower() for word in ["unique", "distinct", "count"]):
        results["unique_counts"] = {col: df[col].nunique() for col in df.columns}
    
    # Data distribution
    if any(word in query.lower() for word in ["distribution", "histogram", "spread"]):
        results["distribution"] = True  # Flag to generate distribution plots
        
    # Column information
    if any(word in query.lower() for word in ["column", "field", "variable"]):
        results["column_info"] = {
            "dtypes": df.dtypes,
            "non_nulls": df.count(),
            "unique_counts": {col: df[col].nunique() for col in df.columns}
        }
    
    # Always include basic info if no specific analysis was triggered
    if not results:
        results["basic_info"] = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes,
            "missing_total": df.isnull().sum().sum()
        }
    
    return results

def generate_visualizations(df: pd.DataFrame, query: str) -> Dict[str, plt.Figure]:
    """Generate visualizations based on the query and dataframe."""
    visualizations = {}
    
    # Correlation heatmap for numeric data
    if "correlation" in query.lower() or "relationship" in query.lower():
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Matrix")
            visualizations["correlation_heatmap"] = fig
    
    # Distribution plots for numeric columns
    if "distribution" in query.lower() or "histogram" in query.lower():
        numeric_cols = df.select_dtypes(include=['number']).columns[:5]  # First 5 numeric columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4*len(numeric_cols)))
            if len(numeric_cols) == 1:
                axes = [axes]  # Make iterable for single column case
            
            for i, col in enumerate(numeric_cols):
                sns.histplot(df[col], kde=True, ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")
            plt.tight_layout()
            visualizations["distributions"] = fig
    
    # Bar charts for categorical columns
    if "count" in query.lower() or "categorical" in query.lower() or "bar" in query.lower():
        cat_cols = df.select_dtypes(include=['object', 'category']).columns[:3]  # First 3 categorical columns
        for i, col in enumerate(cat_cols):
            if df[col].nunique() < 15:  # Only if not too many unique values
                fig, ax = plt.subplots(figsize=(10, 6))
                df[col].value_counts().sort_values(ascending=False).head(10).plot(kind='bar', ax=ax)
                ax.set_title(f"Top 10 values for {col}")
                plt.tight_layout()
                visualizations[f"bar_chart_{col}"] = fig
    
    # Create a simple data overview visualization
    if "overview" in query.lower() or "summary" in query.lower():
        # Data types pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        df.dtypes.value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_title("Column Data Types")
        ax.set_ylabel('')
        visualizations["data_types_pie"] = fig
    
    # If no specific visualizations were triggered but we have numeric data
    if not visualizations and len(df.select_dtypes(include=['number']).columns) > 0:
        numeric_cols = df.select_dtypes(include=['number']).columns[:2]  # First 2 numeric columns
        if len(numeric_cols) == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=ax)
            ax.set_title(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
            visualizations["auto_scatter"] = fig
    
    return visualizations
# Add this function to utils.py
def perform_specific_calculations(df, query):
    """Perform specific calculations based on query keywords."""
    results = {}
    
    # Check for average/mean calculations
    if any(word in query.lower() for word in ["average", "mean", "avg"]):
        # Try to extract column name from query
        col_mentioned = False
        for col in df.columns:
            if col.lower() in query.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    results[f"average_{col}"] = {
                        "value": df[col].mean(),
                        "description": f"Average of {col}: {df[col].mean():.4f}"
                    }
                    col_mentioned = True
                else:
                    results[f"error_{col}"] = f"Cannot calculate average of non-numeric column '{col}'"
                    col_mentioned = True
        
        # If no specific column was mentioned, calculate for all numeric columns
        if not col_mentioned:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                results["average_all"] = {
                    "values": df[numeric_cols].mean().to_dict(),
                    "description": "Average of all numeric columns"
                }
            else:
                results["error"] = "No numeric columns found to calculate average"
    
    # Check for sum calculations
    if any(word in query.lower() for word in ["sum", "total"]):
        # Similar logic for sum as with average
        col_mentioned = False
        for col in df.columns:
            if col.lower() in query.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    results[f"sum_{col}"] = {
                        "value": df[col].sum(),
                        "description": f"Sum of {col}: {df[col].sum():.4f}"
                    }
                    col_mentioned = True
                else:
                    results[f"error_{col}"] = f"Cannot calculate sum of non-numeric column '{col}'"
                    col_mentioned = True
        
        # If no specific column was mentioned, calculate for all numeric columns
        if not col_mentioned:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                results["sum_all"] = {
                    "values": df[numeric_cols].sum().to_dict(),
                    "description": "Sum of all numeric columns"
                }
            else:
                results["error"] = "No numeric columns found to calculate sum"
    
    # Add more calculation types (max, min, count, etc.)
    # ... similar patterns for other calculations
    
    return results