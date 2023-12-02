# Hybrid Feature Selection


## Overview

This repository implements a hybrid feature selection method based on Minimum Redundancy Maximum Relevance (MRMR) and Mutual Information (MI). The goal of feature selection is to identify a subset of the most relevant features while minimizing redundancy among them.

## Methods Implemented

### MRMR-MI Based Feature Selection

The implemented method combines the principles of MRMR and Mutual Information for feature selection. Here's a brief overview:

- **MRMR (Minimum Redundancy Maximum Relevance):** MRMR is an information-theoretic feature selection approach that seeks to maximize the relevance of selected features to the target variable while minimizing redundancy among them.

- **Mutual Information (MI):** Mutual Information is a measure of the statistical dependence between two variables. In the context of feature selection, MI is used to quantify the relationship between each feature and the target variable.

- **Hybrid Approach:** The hybrid feature selection method combines the strengths of MRMR and MI. It leverages MRMR to select features with high relevance to the target and uses MI to further refine the selected features, ensuring a diverse and informative subset.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Available Example
Please, run the examples.ipynb file.

## Available Methods

### MRMR-MI Classification

```python
hybrid_selector = HybridFeatureSelection()
selected_features_classif = hybrid_selector.mrmr_classification(X, y, max_features=10)
```

### MRMR-MI Regression

```python
hybrid_selector = HybridFeatureSelection()
selected_features_regression = hybrid_selector.mrmr_regression(X, y, max_features=10)
```



## References

1. H. Xu, J. Zhang, Y. Lv and P. Zheng, "Hybrid Feature Selection for Wafer Acceptance Test Parameters in Semiconductor Manufacturing," in IEEE Access, vol. 8, pp. 17320-17330, 2020, doi: 10.1109/ACCESS.2020.2966520. [https://ieeexplore.ieee.org/document/8959151]

2. M. Samuel, “MRMR” Explained Exactly How You Wished Someone Explained to You. [https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b]


Still under construction.





