visualize.py
-------------
Author: Saleem Malik

Description:
Visualization utilities for the EduFeatOpt–SmartHive framework.
This module generates:
    1. Feature Importance Plots (SmartFS)
    2. Classifier Performance Comparison (SVM, KNN, BPNN)
    3. Correlation Heatmap of Selected Features
    4. Fitness Progression Plot (ACO + GA)
    5. Boxplots for Metric Distribution

All visualizations are saved in the /results directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure consistent style
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# ============================================================
# Utility: Ensure results directory exists
# ============================================================

def ensure_results_dir():
    if not os.path.exists("results"):
        os.makedirs("results")


# ============================================================
# 1. Plot Feature Importance (from SmartFS)
# ============================================================

def plot_feature_importance(feature_names, scores, top_k=15, save=True):
    """
    Visualizes SmartFS feature importance.
    Parameters:
        feature_names: list of feature names
        scores: list or np.array of importance scores
        top_k: number of top features to show
        save: whether to save plot to /results
    """
    ensure_results_dir()
    df = pd.DataFrame({"Feature": feature_names, "Score": scores})
    df = df.sort_values("Score", ascending=False).head(top_k)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, y="Feature", x="Score", palette="viridis")
    plt.title(f"Top {top_k} SmartFS Feature Importance", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.tight_layout()

    if save:
        plt.savefig("results/smartfs_feature_importance.png")
        plt.close()
    else:
        plt.show()


# ============================================================
# 2. Plot Classifier Performance (from evaluate.py results)
# ============================================================

def plot_classifier_performance(csv_path="results/classifier_performance.csv", save=True):
    """
    Visualizes classifier comparison results from evaluate.py.
    Parameters:
        csv_path: path to saved classifier performance CSV
        save: whether to save output
    """
    ensure_results_dir()
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} not found. Please run evaluate.py first.")
        return

    df = pd.read_csv(csv_path)
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Classifier Performance Comparison", fontsize=14)
    plt.ylim(0, 100)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save:
        plt.savefig("results/classifier_performance_bar.png")
        plt.close()
    else:
        plt.show()


# ============================================================
# 3. Correlation Heatmap (of final selected features)
# ============================================================

def plot_correlation_heatmap(X, feature_names, save=True):
    """
    Displays correlation heatmap of selected features after SmartHive.
    """
    ensure_results_dir()
    corr = pd.DataFrame(X, columns=feature_names).corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, cbar=True)
    plt.title("Correlation Heatmap of Selected Features", fontsize=14)
    plt.tight_layout()

    if save:
        plt.savefig("results/selected_feature_correlation.png")
        plt.close()
    else:
        plt.show()


# ============================================================
# 4. Fitness Progression Plot (ACO + GA)
# ============================================================

def plot_fitness_progression(aco_fitness=None, ga_fitness=None, save=True):
    """
    Plots the fitness improvement across ACO and GA iterations.
    Pass lists or numpy arrays for each stage.
    """
    ensure_results_dir()
    plt.figure(figsize=(8, 5))
    
    if aco_fitness is not None:
        plt.plot(aco_fitness, label="ACO Fitness", marker="o")
    if ga_fitness is not None:
        plt.plot(ga_fitness, label="GA Fitness", marker="s")

    plt.title("SmartHive Fitness Progression (ACO + GA)", fontsize=14)
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Fitness Value")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("results/fitness_progression.png")
        plt.close()
    else:
        plt.show()


# ============================================================
# 5. Boxplot Comparison of Classifier Scores
# ============================================================

def plot_metric_boxplot(csv_path="results/classifier_performance.csv", save=True):
    """
    Shows distribution of classifier metrics (Accuracy, Precision, Recall, F1).
    """
    ensure_results_dir()
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} not found. Please run evaluate.py first.")
        return

    df = pd.read_csv(csv_path)
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.title("Distribution of Classifier Metrics", fontsize=14)
    plt.tight_layout()

    if save:
        plt.savefig("results/metric_boxplot.png")
        plt.close()
    else:
        plt.show()


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    print("\n=== Visualization Module: EduFeatOpt–SmartHive ===")

    # Example usage 1: Feature importance
    try:
        feature_names = np.load("results/final_selected_features.npy", allow_pickle=True)
        scores = np.random.rand(len(feature_names))  # Replace with actual SmartFS scores
        plot_feature_importance(feature_names, scores)
    except Exception as e:
        print("⚠️ Feature importance plot skipped:", e)

    # Example usage 2: Classifier comparison
    plot_classifier_performance()

    # Example usage 3: Boxplot comparison
    plot_metric_boxplot()

    print("\n✅ Visualization complete. Plots saved in /results.")
