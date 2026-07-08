"""
Supervised Machine Learning Module for Project Sehri (The Pure Model).

This module strictly drops the numeric target proxies (aov_score, price_inelasticity_score)
to prevent the critical Data Leakage flaw identified in early architecture drafts.
It then applies SMOTEN (Synthetic Minority Over-sampling Technique for Nominal data)
to safely upscale the severe class imbalance into a 5000:5000 digital twin matrix.
Finally, it trains an L2 (Ridge) Penalized Logistic Regression to extract the
pure predictive drivers of 'high_value_night_intent'.
"""

import pandas as pd
import logging
from typing import Tuple
from imblearn.over_sampling import SMOTEN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def seal_data_leakage(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Enforces the Data Leakage patch by explicitly stripping numeric targets.

    Args:
        df (pd.DataFrame): The Silver Layer containing leaked variables.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The pure categorical feature matrix (X) and target (y).
    """
    logging.info("Applying Data Leakage Patch: Stripping 'aov_score' and 'price_inelasticity_score'.")
    
    # We must drop the variables that inherently define the target
    leakage_columns = ['aov_score', 'price_inelasticity_score', 'Respondent_ID']
    drop_cols = [col for col in leakage_columns if col in df.columns]
    
    y = df['high_value_night_intent']
    X = df.drop(columns=drop_cols + ['high_value_night_intent'])
    
    # Ensure all remaining columns are strings/categories for SMOTEN string parsing
    X = X.astype(str)
    
    logging.info(f"Matrix purified. Remaining Features: {len(X.columns)}")
    return X, y

def generate_digital_twin(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Upscales the extreme minority class utilizing SMOTEN to achieve a balanced 10K matrix.

    Args:
        X (pd.DataFrame): Pure categorical feature matrix.
        y (pd.Series): Binary target.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The perfectly balanced synthetic matrix (5000:5000).
    """
    logging.info(f"Initializing SMOTEN. Original Class Distribution: {y.value_counts().to_dict()}")
    
    # K_neighbors explicitly set to handle the extreme sparsity of the positive class (N=7)
    smoten = SMOTEN(random_state=42, k_neighbors=4)
    X_resampled, y_resampled = smoten.fit_resample(X, y)
    
    logging.info(f"Synthetic Generation Complete. New Class Distribution: {y_resampled.value_counts().to_dict()}")
    return X_resampled, y_resampled

def train_penalized_logit(X_balanced: pd.DataFrame, y_balanced: pd.Series) -> None:
    """
    One-Hot encodes the synthetic matrix, executes the train/test split, 
    and trains an L2 Penalized Logistic Regression to extract pure business drivers.

    Args:
        X_balanced (pd.DataFrame): SMOTEN-upscaled feature matrix.
        y_balanced (pd.Series): SMOTEN-upscaled target.
    """
    logging.info("Executing One-Hot Encoding on nominal features...")
    X_encoded = pd.get_dummies(X_balanced, drop_first=True)
    
    logging.info("Executing 80/20 Train-Test Split on 10,000-row matrix...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    logging.info("Training L2 (Ridge) Penalized Logistic Regression...")
    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    logging.info("\n--- PURE MODEL EVALUATION (TEST SET N=2000) ---")
    logging.info("\n" + classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info("Pure ML Pipeline execution complete.")

def run_propensity_pipeline(df_silver: pd.DataFrame) -> None:
    """
    Orchestrates the Supervised ML workflow.

    Args:
        df_silver (pd.DataFrame): The validated Silver Layer data.
    """
    X, y = seal_data_leakage(df_silver)
    X_sm, y_sm = generate_digital_twin(X, y)
    train_penalized_logit(X_sm, y_sm)

if __name__ == "__main__":
    # Dynamically locate the root directory regardless of where the repo is cloned
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "sehri_silver_features.parquet")
    
    if os.path.exists(SILVER_PATH):
        df_valid = pd.read_parquet(SILVER_PATH)
        run_propensity_pipeline(df_valid)
    else:
        logging.error("CRITICAL: Silver Layer not found. Run validation/ETL first.")