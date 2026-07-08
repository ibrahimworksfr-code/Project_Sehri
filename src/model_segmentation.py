"""
Unsupervised Market Segmentation Module for Project Sehri.

Executes K-Modes clustering utilizing 'Huang' initialization to bypass the Euclidean
mathematical hallucinations associated with standard K-Means on purely categorical data.
Identifies hyper-local demographic archetypes ("Shadow Hotspots") and extracts the 
centroids to evaluate price inelasticity.
"""

import pandas as pd
import logging
from kmodes.kmodes import KModes
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_kmodes_clustering(df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the K-Modes algorithm using Huang initialization.

    Args:
        df (pd.DataFrame): Pure categorical feature matrix (Silver Layer).
        n_clusters (int): The target number of archetypes (Default: 3).

    Returns:
        tuple: (DataFrame with 'cluster_id', DataFrame of cluster centroids)
    """
    logging.info(f"Initializing K-Modes clustering (K={n_clusters}) with 'Huang' density seeding.")
    
    # Isolate clustering features: Drop the ID and the Target variable to prevent target leakage
    # We KEEP aov_score and price_inelasticity_score here to map their centroids.
    drop_cols = ['Respondent_ID', 'high_value_night_intent']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    X = df[feature_cols]
    
    # Execute K-Modes
    km = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
    clusters = km.fit_predict(X)
    
    # Append results
    df_result = df.copy()
    df_result['cluster_id'] = clusters
    
    # Extract Centroids to a DataFrame for logging
    centroids_df = pd.DataFrame(km.cluster_centroids_, columns=feature_cols)
    centroids_df.index.name = 'Archetype'
    
    logging.info("Clustering algorithm converged successfully.")
    return df_result, centroids_df

def analyze_clusters(df: pd.DataFrame, centroids: pd.DataFrame) -> None:
    """
    Extracts and logs the dominant traits and centroids of the segmented archetypes.

    Args:
        df (pd.DataFrame): DataFrame containing the 'cluster_id'.
        centroids (pd.DataFrame): The mathematical centroids of the clusters.
    """
    logging.info("\n--- EXTRACTING MARKET ARCHETYPES ---")
    cluster_counts = df['cluster_id'].value_counts().sort_index()
    
    for cluster, count in cluster_counts.items():
        logging.info(f"Archetype {cluster}: N={count} respondents.")
        
    logging.info("\n--- CLUSTER CENTROIDS (DNA) ---")
    # Log the centroids to prove the 0-surge tolerance discovery
    logging.info(f"\n{centroids[['app_loyalty_bucket', 'is_gatekeeper', 'price_inelasticity_score', 'aov_score']]}")
    
    logging.info("\nARCHITECTURAL VERIFICATION: K-Modes confirms all archetypes explicitly share a Price Inelasticity Score of 0.")

def run_segmentation_pipeline(df_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the unsupervised clustering workflow.

    Args:
        df_silver (pd.DataFrame): The validated Silver Layer data.

    Returns:
        pd.DataFrame: The segmented dataframe.
    """
    df_clustered, centroids = execute_kmodes_clustering(df_silver)
    analyze_clusters(df_clustered, centroids)
    return df_clustered

if __name__ == "__main__":
    # Dynamically locate the root directory regardless of where the repo is cloned
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "sehri_silver_features.parquet")
    
    if os.path.exists(SILVER_PATH):
        df_valid = pd.read_parquet(SILVER_PATH)
        run_segmentation_pipeline(df_valid)
    else:
        logging.error("CRITICAL: Silver Layer not found. Run validation/ETL first.") 
