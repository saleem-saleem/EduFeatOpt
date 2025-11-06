Smart Filter Feature Selection Algorithm (SmartFS)
---------------------------------------------------
Author: Saleem Malik
Description:
A hybrid filter-based feature selection framework that combines
Chi-Square, ReliefF, and Correlation-based Feature Selection (CFS)
to identify the most relevant and non-redundant features.

Datasets used: Student Performance Dataset (UCI) or any tabular dataset.
"""

# Required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Optional: ReliefF implementation
# Install ReliefF if needed: pip install skrebate
from skrebate import ReliefF

# ============================================================
# STEP 1: Data Preprocessing
# ============================================================

def preprocess_data(df, target_column):
    """
    Encodes categorical columns, scales numerical columns,
    and separates features and target.
    """
    df = df.copy()
    label_encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns


# ============================================================
# STEP 2: Chi-Square Feature Selection
# ============================================================

def chi_square_selection(X, y, top_k):
    """
    Uses Chi-Square test to select top K relevant features.
    """
    chi_selector = SelectKBest(score_func=chi2, k=top_k)
    chi_selector.fit(X, y)
    scores = chi_selector.scores_
    return scores


# ============================================================
# STEP 3: ReliefF Feature Selection
# ============================================================

def reliefF_selection(X, y):
    """
    Uses ReliefF algorithm to evaluate feature importance.
    """
    relief = ReliefF(n_neighbors=10, n_jobs=-1)
    relief.fit(X, y)
    return relief.feature_importances_


# ============================================================
# STEP 4: Correlation-based Feature Selection (CFS)
# ============================================================

def correlation_based_selection(X, y):
    """
    Computes Correlation-based Feature Selection (CFS) score.
    A feature is strong if it's correlated with the target but
    weakly correlated with other features.
    """
    df = pd.DataFrame(X)
    corr_matrix = df.corr()
    target_corr = np.array([np.corrcoef(df.iloc[:, i], y)[0, 1] for i in range(df.shape[1])])
    cfs_scores = []

    for i in range(df.shape[1]):
        numerator = abs(target_corr[i])
        denominator = np.sum(np.abs(corr_matrix.iloc[i, :])) / df.shape[1]
        cfs_score = numerator / (denominator + 1e-6)
        cfs_scores.append(cfs_score)

    return np.array(cfs_scores)


# ============================================================
# STEP 5: SmartFS Aggregation Scoring Method
# ============================================================

def aggregate_scores(chi_scores, relief_scores, cfs_scores):
    """
    Normalizes and combines Chi-Square, ReliefF, and CFS scores
    into a single composite feature importance score.
    """
    def normalize(arr):
        arr = np.nan_to_num(arr)
        return (arr - np.min(arr)) / (np.ptp(arr) + 1e-6)

    chi_norm = normalize(chi_scores)
    relief_norm = normalize(relief_scores)
    cfs_norm = normalize(cfs_scores)

    # Weighted combination (weights can be tuned experimentally)
    final_score = 0.4 * chi_norm + 0.3 * relief_norm + 0.3 * cfs_norm
    return final_score


# ============================================================
# STEP 6: SmartFS Main Function
# ============================================================

def SmartFS(df, target_column, top_k=10):
    """
    Executes Smart Filter Feature Selection process.
    Returns the top K most relevant features.
    """
    X, y, feature_names = preprocess_data(df, target_column)

    print("Computing Chi-Square scores...")
    chi_scores = chi_square_selection(X, y, top_k)

    print("Computing ReliefF scores...")
    relief_scores = reliefF_selection(X, y)

    print("Computing CFS scores...")
    cfs_scores = correlation_based_selection(X, y)

    print("Aggregating SmartFS scores...")
    final_scores = aggregate_scores(chi_scores, relief_scores, cfs_scores)

    ranked_features = sorted(
        list(zip(feature_names, final_scores)),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop {} Selected Features:".format(top_k))
    for feat, score in ranked_features[:top_k]:
        print(f"{feat:25s} -> Score: {score:.4f}")

    selected_features = [feat for feat, _ in ranked_features[:top_k]]
    return selected_features, final_scores


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Example: Load Student Performance Dataset (from UCI)
    df = pd.read_csv("data/student-mat.csv")   # or your dataset
    target_column = "G3"  # final grade column

    top_features, scores = SmartFS(df, target_column, top_k=10)
