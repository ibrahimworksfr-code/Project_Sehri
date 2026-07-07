"""
Hypothesis Testing Module for Project Sehri.

Executes rigorous statistical inference on the Silver Layer categorical data to validate
the "Ramadan Sine Wave" temporal hypotheses (H1, H2, H3).
Implements Yates-Corrected Chi-Square to safely handle low cell frequencies typical of 
hyper-local surveys, maintaining strict parity with EXP_01.
"""

import pandas as pd
import logging
from scipy.stats import chi2_contingency
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_yates_chi_square(contingency_table: pd.DataFrame, hypothesis_name: str) -> None:
    """
    Executes a Yates-Corrected Chi-Square test on a 2x2 contingency table.

    Args:
        contingency_table (pd.DataFrame): The observed frequency matrix dynamically generated from data.
        hypothesis_name (str): Identifier for logging outputs.
    """
    logging.info(f"--- Testing {hypothesis_name} ---")
    logging.info(f"Observed Matrix:\n{contingency_table}")
    
    # Applying Yates' correction (correction=True)
    chi2, p_val, dof, expected = chi2_contingency(contingency_table, correction=True)
    
    logging.info(f"Chi-Square Statistic (Yates): {chi2:.4f} | P-Value: {p_val:.4f}")
    
    if p_val < 0.05:
        logging.info("Verdict: VALIDATED (Statistically Significant)")
    else:
        logging.info("Verdict: REJECTED (Null Wins - Directional Signal Only if applicable)")
    logging.info("-" * 40)

def run_hypothesis_pipeline(df_silver: pd.DataFrame) -> None:
    """
    Orchestrates the hypothesis testing sequence directly from the Silver matrix.

    Args:
        df_silver (pd.DataFrame): The validated Silver Layer data.
    """
    logging.info("Initializing Hypothesis Testing Sequence.")
    
    # H1: THE SEHRI SPIKE (Night Volume)
    h1_crosstab = pd.crosstab(df_silver['is_ramadan'], df_silver['time_night'])
    execute_yates_chi_square(h1_crosstab, "H1 (The Sehri Spike: 10 PM - 2 AM)")

    # H2: THE DAYTIME VALLEY (Afternoon Drop-off)
    h2_crosstab = pd.crosstab(df_silver['is_ramadan'], df_silver['time_afternoon'])
    execute_yates_chi_square(h2_crosstab, "H2 (The Daytime Valley: 12 PM - 4 PM)")
    
    # H3: THE IFTAR PANIC (Evening Spike)
    h3_crosstab = pd.crosstab(df_silver['is_ramadan'], df_silver['time_evening'])
    execute_yates_chi_square(h3_crosstab, "H3 (The Iftar Panic: 5 PM - 9 PM)")

if __name__ == "__main__":
    # Absolute pathing for standalone module testing
    PROJECT_ROOT = r"C:\Node2_Workspace\03_Projects\Project_Sehri"
    SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "sehri_silver_features.parquet")
    
    if os.path.exists(SILVER_PATH):
        df = pd.read_parquet(SILVER_PATH)
        run_hypothesis_pipeline(df)
    else:
        logging.error("CRITICAL: Silver Layer not found. Run validation/ETL first.")