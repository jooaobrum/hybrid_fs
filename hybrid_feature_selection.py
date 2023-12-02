from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class MRMRFeatureSelection:
    def __init__(self):
        self.selected_features = []

    def calculate_relevance_classif(self, X, Y):
        n_features = X.shape[1]
        relevance_values = np.zeros(n_features)

        X = X.values

        for i in range(n_features):
            relevance_values[i] = mutual_info_classif(X[:, i].reshape(-1, 1), Y, random_state=42)[0]

        return relevance_values

    def calculate_relevance_regression(self, X, Y):
        n_features = X.shape[1]
        relevance_values = np.zeros(n_features)

        X = X.values

        for i in range(n_features):
            relevance_values[i] = mutual_info_regression(X[:, i].reshape(-1, 1), Y, random_state=42)[0]

        return relevance_values

    def calculate_redundancy(self, X, actual_feature, selected_features):
        n_selected_features = len(selected_features)
        redundancy_values = np.zeros(n_selected_features)

        for i, feat in enumerate(selected_features):
            redundancy_values[i] = mutual_info_regression(
                X.loc[:, actual_feature].values.reshape(-1, 1),
                X.loc[:, feat].values.reshape(-1, 1),
                random_state=42
            )[0]

        return redundancy_values

    def mrmr_classification(self, X, y, max_features=50):
        features = X.columns.tolist()
        n_features = len(features)
        K = min(max_features, n_features)
        not_selected_features = features
        self.selected_features = []

        for i in tqdm(range(K), disable=True):
            relevance = self.calculate_relevance_classif(X.loc[:, not_selected_features], y.values.ravel())

            if i > 0:
                last_selected_feature = self.selected_features[-1]
                redundancy = self.calculate_redundancy(X, last_selected_feature, not_selected_features)
            else:
                redundancy = np.zeros(len(not_selected_features))

            s_term = 1 / len(self.selected_features) if len(self.selected_features) > 0 else 1
            current_J = relevance - s_term * redundancy

            best_feature = not_selected_features[np.argmax(current_J)]
            self.selected_features.append(best_feature)
            not_selected_features.remove(best_feature)

        return self.selected_features

    def mrmr_regression(self, X, y, max_features=50):
        features = X.columns.tolist()
        n_features = len(features)
        K = min(max_features, n_features)
        not_selected_features = features
        self.selected_features = []

        for i in tqdm(range(K)):
            relevance = self.calculate_relevance_regression(X.loc[:, not_selected_features], y.values.ravel())

            if i > 0:
                last_selected_feature = self.selected_features[-1]
                redundancy = self.calculate_redundancy(X, last_selected_feature, not_selected_features)
            else:
                redundancy = np.zeros(len(not_selected_features))

            s_term = 1 / len(self.selected_features) if len(self.selected_features) > 0 else 1
            current_J = relevance - s_term * redundancy

            best_feature = not_selected_features[np.argmax(current_J)]
            self.selected_features.append(best_feature)
            not_selected_features.remove(best_feature)

        return self.selected_features