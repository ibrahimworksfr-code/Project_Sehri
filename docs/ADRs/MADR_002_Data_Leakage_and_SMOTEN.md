# MADR-002: Resolving Data Leakage and Downgrading to SMOTEN

## Status

Accepted

## Date

2026-07-05

## Context

During the initial deployment of the predictive pipeline, the logistic regression model achieved unusually high recall scores (approaching 94%). An audit of the feature importance coefficients revealed that the model was relying heavily on two numeric proxies: `aov_score` (Average Order Value) and `price_inelasticity_score` (tolerance for surge fees).
Because the binary target variable `high_value_night_intent` is inherently defined by a user's willingness to spend high amounts and tolerate surge pricing, including these numeric proxies as training features created a severe **Data Leakage** loop. The model was effectively given the answer key during training. 

## Decision

To seal the data leakage, we executed a hard feature purge. We explicitly stripped all continuous numeric targets (`aov_score` and `price_inelasticity_score`) from the training feature matrix.
By removing the only continuous variables, the dataset transformed into a **100% purely categorical (nominal) matrix**. Consequently, we downgraded the synthetic upscaling engine from SMOTENC (designed for mixed categorical and continuous data) to **SMOTEN (Synthetic Minority Over-sampling Technique for Nominal data)**. 

## Consequences

* **Positive**: Stripping the continuous variables mathematically sealed the data leakage, forcing the model to learn actual behavioral and demographic patterns rather than cheating via definitional proxies.
* **Positive**: Transitioning to SMOTEN ensured that the synthesizer exclusively utilized algorithms optimized for pure categorical distributions, preventing potential execution errors that arise when SMOTENC attempts to process arrays devoid of continuous columns.
* **Negative**: The model's raw accuracy/recall metrics naturally decreased compared to the leaked version, but the resulting predictions now represent valid, generalizable business intelligence.

## Rejected Alternatives

### 1. Retaining SMOTENC on a Pure Categorical Matrix
* **Why it was rejected**: SMOTENC is hardcoded to expect at least one continuous variable to calculate its synthetic midpoints. Feeding it a matrix stripped of all continuous features either triggers an immediate compiler exception or forces the algorithm to mathematically fail, generating corrupted digital twins. SMOTEN was the only architecturally sound choice post-purge.
