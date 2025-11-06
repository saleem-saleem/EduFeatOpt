main.py
-------
Author: Saleem Malik

Description:
Main driver script for the EduFeatOpt-SmartHive Framework.

This script executes the complete feature optimization and
evaluation pipeline:
    1. SmartFS — Filter-based feature selection (Chi2, ReliefF, CFS)
    2. SmartHive — Hybrid ACO–GA wrapper optimization with SVM fitness
    3. Evaluation — SVM, KNN, and BPNN classification metrics

Outputs:
    - Selected feature list
    - Accuracy metrics (CSV)
    - Confusion matrices & plots
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Import modules (ensure they’re in the same folder)
from smartfs import SmartFS
from smarthive import SmartHive
from evaluate import evaluate_models


# ============================================================
# Helper Functions
# ============================================================

def ensure_directories():
    """Ensure that required folders exist."""
    for folder in ["data", "results"]:
        if not os.path.exists(folder):
            os.makedirs(folder)


def encode_categorical(df):
    """Encode categorical features in dataset."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(dataset_path="data/student-mat.csv", target_col="G3", top_k=20):
    """
    Runs the complete EduFeatOpt-SmartHive feature selection pipeline.
    """
    ensure_directories()

    print("\n=== 🚀 EDUFEATOPT SMART HIVE FRAMEWORK STARTED ===")

    # ------------------------------------------------------------
    # STEP 1: Load and preprocess dataset
    # ------------------------------------------------------------
    print(f"\n📂 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df = encode_categorical(df)

    print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

    # ------------------------------------------------------------
    # STEP 2: SmartFS Filter-based Feature Selection
    # ------------------------------------------------------------
    print("\n🔍 Running SmartFS (Chi2 + ReliefF + CFS)...")
    selected_features, _ = SmartFS(df, target_col, top_k=top_k)
    print(f"\n✅ SmartFS Selected Top {len(selected_features)} Features:")
    print(selected_features)

    # Prepare filtered dataset
    df_filtered = df[selected_features + [target_col]]
    X_filtered = df_filtered.drop(columns=[target_col]).values
    y = df_filtered[target_col].values

    # Scale features for optimization
    scaler = MinMaxScaler()
    X_filtered = scaler.fit_transform(X_filtered)

    # Save intermediate files
    np.save("results/X_filtered.npy", X_filtered)
    np.save("results/y.npy", y)

    # ------------------------------------------------------------
    # STEP 3: SmartHive Wrapper Optimization (ACO + GA)
    # ------------------------------------------------------------
    print("\n🐝 Running SmartHive (ACO + GA Optimization with SVM Fitness)...")
    selected_idx, best_fit = SmartHive(X_filtered, y)
    X_selected = X_filtered[:, selected_idx]

    print(f"\n✅ SmartHive Optimization Complete — Best Fitness: {best_fit:.4f}")
    print(f"Selected {len(selected_idx)} features after optimization.")

    # Save selected feature results
    np.save("results/X_selected.npy", X_selected)
    np.save("results/smarthive_selected_indices.npy", selected_idx)
    np.save("results/smarthive_best_fitness.npy", best_fit)

    selected_final_features = [selected_features[i] for i in selected_idx]
    with open("results/final_selected_features.txt", "w") as f:
        for feat in selected_final_features:
            f.write(feat + "\n")

    print(f"💾 Final selected feature names saved to results/final_selected_features.txt")

    # ------------------------------------------------------------
    # STEP 4: Model Evaluation
    # ------------------------------------------------------------
    print("\n📊 Evaluating SVM, KNN, and BPNN performance...")
    metrics_df = evaluate_models(X_selected, y)
    print("\n✅ Evaluation Completed! Results saved in /results directory.")

    print("\n=== 🎯 EDUFEATOPT SMART HIVE FRAMEWORK FINISHED SUCCESSFULLY ===")
    return metrics_df


# ============================================================
# Run Main
# ============================================================

if __name__ == "__main__":
    # Example: Student Performance Dataset (UCI)
    dataset_path = "data/student-mat.csv"
    target_col = "G3"   # Final grade

    results = run_pipeline(dataset_path, target_col, top_k=20)
