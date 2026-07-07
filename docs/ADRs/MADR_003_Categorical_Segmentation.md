# MADR-003: Categorical Segmentation via K-Modes

## Status

Accepted

## Date

2026-07-05

## Context

Following the Supervised Learning phase, the pipeline required an Unsupervised Machine Learning phase to segment the $N=131$ respondent pool into distinct market archetypes (identifying "Shadow Hotspots" without a predefined target variable). 
Because the data leakage patch (MADR-002) completely stripped the matrix of numeric variables, the remaining feature stack consists entirely of nominal text strings. Standard clustering paradigms rely heavily on Euclidean geometry, which is mathematically incompatible with pure categorical data. 

## Decision

We deployed the **K-Modes** clustering algorithm utilizing **Hamming Distance**:

$$ \text{Hamming Dissimilarity}(X, Y) = \sum_{j=1}^{m} \delta(x_j, y_j) $$

Where $\delta(x_j, y_j) = 0$ if attributes match perfectly, and $1$ if they differ. Hamming distance quantifies dissimilarity based exclusively on the total count of categorical feature mismatches between observation vectors. The algorithm iteratively updates centroids by identifying the statistical "mode" (the most frequent categorical value) rather than a mathematical average. To ensure reproducible stability, the engine was explicitly initialized using the 'Huang' method.

## Consequences

* **Positive**: K-Modes mathematically aligns with the pure categorical structure, successfully identifying three distinct archetypes (Family Cart Maximizer, Zepto Household Routine, Solo Scroller) without corrupting text variables.
* **Positive**: K-Modes operates with high computational efficiency, avoiding the massive memory overhead required to hold full pre-computed distance matrices in RAM.
* **Negative**: The algorithm forces a predefined number of clusters ($K=3$), requiring business intuition to set the parameter upfront, unlike agglomerative methods.

## Rejected Alternatives

### 1. Standard K-Means Clustering
* **Why it was rejected**: K-Means minimizes variance by calculating the Euclidean distance (continuous mathematical mean) between points. It is fundamentally impossible to calculate the average of text strings like "Tolichowki" and "Banjara Hills." Attempting to force one-hot encoded text into a Euclidean engine results in severe mathematical hallucinations.

### 2. Gower's Distance and Agglomerative Hierarchical Clustering
* **Why it was rejected**: Gower's Distance is a brilliant coefficient mathematically designed to handle *mixed* data types (combining continuous numbers with categorical strings). However, because MADR-002 removed all numeric features, the matrix was 100% categorical. On pure categorical data, Gower's calculation simplifies down to exact Hamming Distance logic. Deploying Agglomerative Clustering on top of a Gower's matrix in this scenario introduces unnecessary computational bloat and memory overhead ($N \times N$ matrix calculation) without providing any additional business intelligence over K-Modes.
