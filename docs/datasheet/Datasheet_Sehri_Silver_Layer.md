# Datasheet: Project Sehri Silver Layer ($N=131$)

## 1. Dataset Overview
This datasheet documents the immutable "Silver Layer" of the Project Sehri dataset, representing the raw, cleaned survey responses from $N=131$ participants. This dataset serves as the absolute Ground Truth for all hypothesis testing and baseline modeling.

## 2. Data Collection & Methodology
- **Instrument**: Google Forms Questionnaire (16 Questions).
- **Sampling Method**: Non-Probability Convenience Sampling.
- **Target Population**: Residents of Hyderabad utilizing Quick-Commerce apps (Zepto, Blinkit, Swiggy Instamart).

## 3. Sampling Bias and Limitations
During the exploratory data analysis phase, severe limitations were identified in the sampling methodology:
- **Convenience Sampling Skew**: The survey distribution heavily circulated among university students and young professionals.
- **Undercoverage Bias**: The target demographic required to validate a profitable night economy (the "Family Gatekeeper" in the Old City buying high-AOV bulk items) was severely under-represented. 
- **Resulting Imbalance**: The dataset exhibited a catastrophic class imbalance regarding the `high_value_night_intent` target, with 123 negative instances and only 7 positive instances. This bias caused early hypothesis tests (like the Daytime Valley) to fail mathematically because budget-conscious students (who refuse surge fees) dominated the sample.

## 4. Data Governance and Lineage Protocol
To preserve strict data integrity and MLOps best practices:
- **The Silver Layer is Immutable**: The $N=131$ dataset (`sehri_silver_features.parquet`) is the *only* data state saved to the local disk in the `processed/` directory.
- **Synthetic Data is a Compute State**: The 10,000 synthetic rows generated during modeling (via SMOTENC and SMOTEN) are explicitly defined as a **Volatile Compute State**. They are generated Just-In-Time (JIT) in RAM during pipeline execution and flushed post-training. 
- **Rule**: Never serialize or store synthetic digital twins on the local hard drive. Storing fake data pollutes the data lineage and invalidates audit trails.
