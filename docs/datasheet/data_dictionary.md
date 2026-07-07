# Data Dictionary: Project Sehri (Medallion Architecture)

## Executive Summary
This document delineates the data lineage from the raw ingestion (Bronze Layer) through the cleaning and feature engineering phase (Silver Layer), to the highly optimized, immutable state ready for machine learning consumption (Gold Layer). It explicitly defines the feature constraints, mappings, and null-handling protocols established within the `01_eda_night_mode.ipynb` refinery pipeline.

## Data Dictionary & Lineage Mapping

| Feature Name | Data Type | Target Layer | Business & Technical Definition | Constraints/Null Handling |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | String | Bronze | Exact UTC timestamp of survey submission via Google Forms. | None. N/A for modeling. |
| `q0_gender` | Categorical | Silver | Self-identified gender cohort. | Validated against string variations; nulls dropped. |
| `q1_age` | Categorical | Silver | Demographic age bucket (e.g., Gen Z / College, Young Pro). | Standardized mapping applied. |
| `q2_occupation` | Categorical | Silver | Current professional status (e.g., Student, Homemaker, Working Professional). | Standardized text mapping. |
| `q3_area` | Categorical | Silver | Micro-geographical cluster within Hyderabad (e.g., Tolichowki, Barkas). | Consolidated into `macro_zone` feature to prevent high-cardinality dimensionality explosion. |
| `macro_zone` | Categorical | Silver Engineered | Geofenced routing hub classification (e.g., Old City & South). Derived from `q3_area`. | Deterministic mapping; unmapped outliers handled via 'Other' bucket. |
| `q4_income` | Categorical | Silver | Monthly disposable income range. | Categorical binning. |
| `q5_primary_app` | Categorical | Silver | Q-Commerce platform capturing the highest wallet share (Blinkit, Zepto, Swiggy Instamart). | Parsed from multi-select arrays. |
| `app_loyalty_bucket` | Categorical | Silver Engineered | Classifies users as Platform Loyalists vs. App Agnostic based on multi-platform usage. | Derived via boolean logic on `q5_primary_app`. |
| `q6_order_freq` | Categorical | Silver | Baseline purchasing frequency velocity. | Categorical standardizations. |
| `q7_aov` | Categorical | Silver | Average Order Value per checkout. | Mapped to `aov_score` ordinal ranking for predictive modeling. |
| `aov_score` | Integer | Silver Engineered | Ordinal mapping of `q7_aov` (e.g., 0, 1, 2) to establish baseline profitability potential. | Deterministic cast to integer. |
| `q8_order_time` | Categorical | Silver | Temporal consumption patterns (Daytime Valley vs. Sehri Spike). | One-Hot Encoded into Boolean flags for Hypotheses testing. |
| `Items_Intent_Q9` (Mapped to `q9_late_night_items`) | Categorical | Silver | The core inventory vector characterizing the graveyard shift demand. **Crucial Mapping:** This feature categorizes product intent. By mathematically correlating this array against actual profitability, the pipeline explicitly refutes the "Milk Index" fallacy. The data definitively proves that low-margin staples (milk, bread) generate negative unit economics during night operations. Conversely, intent signals for premium items (e.g., imported dates, frozen gourmet kebabs, Haleem) validate the transition to High-AOV bundles (Sehri Power Kits). | String parsing applied to multi-select responses. Nulls safely assumed as zero intent for night orders. |
| `q10_primary_reason` | Categorical | Silver | The core utility driver (Speed vs. Comfort vs. Discounts) overriding local offline Kirana stores. | Text standardization. |
| `q11_delivery_fee_tolerance` | Categorical | Silver | Evaluates consumer elasticity against operational surge pricing. | Correlated with `price_inelasticity_score`. |
| `price_inelasticity_score` | Integer | Silver Engineered | Ordinal ranking of surge fee tolerance (0 = highly sensitive, >0 = inelastic). | Deterministic derivation from Q11. |
| `q12_sat_speed` / `q12_sat_quality` / `q12_sat_ui` / `q12_sat_support` | Integer | Silver | Customer Satisfaction (CSAT) scores on a 1-5 Likert scale across operational dimensions. | Safely cast to numerical types; missing values imputed via mode. |
| `q13_switch_speed` | Boolean | Silver | Indicates platform switching behavior driven by latency (Iftar Panic window). | Binarized (Yes=1, No=0). |
| `q14_switch_price` | Boolean | Silver | Indicates platform switching driven by promotional subsidization. | Binarized (Yes=1, No=0). |
| `Household_Ctx_Q16` (Derived from Q15/Q16 context) | Categorical | Silver | Evaluates whether the user is ordering for a household/family unit or as a solo consumer. **Crucial Mapping:** When clustered alongside High-AOV intents, this feature defines the "Family Gatekeeper" archetype. The pipeline establishes that solo consumers (students) ordering low-margin snacks destroy nocturnal margins, while the Family Gatekeeper executing bulk orders for the household subsidizes the graveyard shift's operational friction. | Parsed and mapped into the binary `is_gatekeeper` feature. Nulls logically inferred as solo consumers (0) unless contradictory AOV exists. |
| `is_gatekeeper` | Boolean | Silver Engineered | Binary flag establishing whether the user acts as a procurement proxy for a larger household (1) or a solo consumer (0). | Derived deterministically from household context inputs. |
| `is_ramadan` | Boolean | Silver | Denotes whether the observation belongs to the Pre-Ramadan baseline or the During-Ramadan active test group. | Binarized control variable (0 = Baseline, 1 = Test). |
| `high_value_night_intent` | Boolean | Gold | The master target variable for supervised propensity modeling (Firth's Logistic Regression). Identifies cohorts highly likely to purchase premium bundles during the Sehri graveyard shift. | Engineered target. Strictly binarized for classification. |
