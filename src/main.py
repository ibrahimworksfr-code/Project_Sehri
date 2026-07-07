"""
Master Orchestrator for Project Sehri MLOps Pipeline.

This script executes the entire pipeline sequentially:
1. Data Validation (Ingesting the immutable Silver Layer)
2. Hypothesis Testing (Inferential Statistics)
3. Market Segmentation (Unsupervised K-Modes)
4. Propensity Modeling (Supervised Pure L2 Logit via SMOTEN)
"""

import os
import logging
import sys

# Import the modular components
from data_preprocessing import load_and_validate_silver_layer
from hypothesis_testing import run_hypothesis_pipeline
from model_segmentation import run_segmentation_pipeline
from model_propensity import run_propensity_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ProjectSehri_Orchestrator")

def execute_pipeline() -> None:
    """
    Executes the Project Sehri pipeline sequentially.
    """
    logger.info("="*50)
    logger.info("INITIALIZING PROJECT SEHRI MLOPS PIPELINE")
    logger.info("="*50)
    
    # Define Strict Absolute Paths
    PROJECT_ROOT = r"C:\Node2_Workspace\03_Projects\Project_Sehri"
    SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "sehri_silver_features.parquet")
    
    # Phase 1: Data Validation
    logger.info("\n--- PHASE 1: DATA VALIDATION (SILVER LAYER) ---")
    df_silver = load_and_validate_silver_layer(SILVER_PATH)
    
    # Phase 2: Hypothesis Testing
    logger.info("\n--- PHASE 2: HYPOTHESIS TESTING (INFERENTIAL STATISTICS) ---")
    run_hypothesis_pipeline(df_silver)
    
    # Phase 3: Unsupervised Market Segmentation
    logger.info("\n--- PHASE 3: K-MODES MARKET SEGMENTATION ---")
    run_segmentation_pipeline(df_silver)
    
    # Phase 4: Supervised Propensity Modeling
    logger.info("\n--- PHASE 4: PURE PROPENSITY MODELING (SMOTEN + L2 LOGIT) ---")
    run_propensity_pipeline(df_silver)
    
    logger.info("="*50)
    logger.info("PIPELINE EXECUTION COMPLETE.")
    logger.info("="*50)

if __name__ == "__main__":
    execute_pipeline()
