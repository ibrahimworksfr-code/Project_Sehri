"""
Data Preprocessing & Validation Module for Project Sehri.

Responsible for ingesting the immutable Silver Layer (Parquet), validating schema integrity, 
and ensuring no data corruption occurred prior to passing the matrix to the ML pipelines.
"""

import pandas as pd
import os
import logging
from typing import List

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_validate_silver_layer(filepath: str) -> pd.DataFrame:
    """
    Ingests the pre-processed Silver Layer and validates its schema.

    Args:
        filepath (str): Absolute path to the Silver Layer Parquet file.

    Returns:
        pd.DataFrame: The validated Silver Layer matrix.
        
    Raises:
        FileNotFoundError: If the Parquet file is missing.
        ValueError: If mandatory schema columns are missing.
    """
    logging.info(f"Initiating Silver Layer ingestion from: {filepath}")
    
    if not os.path.exists(filepath):
        logging.error("CRITICAL FAILURE: Immutable Silver Layer not found.")
        raise FileNotFoundError(f"Missing required artifact at {filepath}. Ensure ETL notebook 01 was executed.")

    df = pd.read_parquet(filepath)
    logging.info(f"Silver Layer loaded successfully. Shape: {df.shape}")
    
    # Define the strict enterprise schema expected for the models
    mandatory_schema: List[str] = [
        'Respondent_ID', 'macro_zone', 'app_loyalty_bucket', 
        'is_ramadan', 'is_gatekeeper', 'price_inelasticity_score', 'aov_score', 
        'time_afternoon', 'time_evening', 'time_night', 'high_value_night_intent'
    ]
    
    # Audit for missing columns
    missing_cols = [col for col in mandatory_schema if col not in df.columns]
    if missing_cols:
        logging.error(f"SCHEMA BREACH: Missing critical features: {missing_cols}")
        raise ValueError(f"Silver Layer corrupted. Missing features: {missing_cols}")
        
    logging.info("Schema validation passed. All mandatory features present.")
    
    # Audit for NULL values
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        logging.warning(f"DATA INTEGRITY WARNING: Found {null_counts} NaN values in the Silver Layer.")
    else:
        logging.info("Data integrity verified. Matrix contains zero null values.")

    return df

if __name__ == "__main__":
    # Test execution
    PROJECT_ROOT = r"C:\Node2_Workspace\03_Projects\Project_Sehri"
    SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "sehri_silver_features.parquet")
    
    try:
        df_silver = load_and_validate_silver_layer(SILVER_PATH)
        print("\n=== ETL VALIDATION SUCCESS ===")
        print(df_silver.head())
    except Exception as e:
        print(f"\n=== ETL VALIDATION FAILED ===\n{e}")