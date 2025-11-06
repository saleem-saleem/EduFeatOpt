evaluate.py
------------
Author: Saleem Malik

Description:
Evaluates classification performance of features selected using
SmartFS (filter-based) and SmartHive (ACO–GA wrapper-based)
feature selection algorithms.

This script tests multiple classifiers (SVM, KNN, BPNN)
and reports metrics like Accuracy, Precision, Recall, and F1-score.

It can be used standalone or as a post-processing step in the
EduFeatOpt framework pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# Utility Functions
# ============================================================

def ensure_results_dir():
    """Ensure results directory exists."""
    if not os.path.exists("results"):
        os.makedirs("results")


def print_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_confusion_matrix.png")
    plt.close()


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_models(X, y, test_size=0.2, random_state=42, save_results=True):
    """
    Evaluate selected features using multiple classifiers:
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Backpropagation Neural Network (BPNN / MLP)

    Parameters:
        X: numpy array, feature matrix
        y: numpy array, class labels
        test_size: fraction for test split
        random_state: random seed for reproducibility
        save_results: whether to save results to CSV and plots

    Returns:
        Pandas DataFrame containing performance metrics
    """

    ensure_results_dir()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Define models
    models = {
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "BPNN": MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=random_state)
    }

    # Initialize results dictionary
    results = []

    # Evaluate each model
    for name, model in models.items():
        print(f"\n🚀 Training {name} classifier...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        results.append({
            "Model": name,
            "Accuracy": round(acc * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1-Score": round(f1 * 100, 2)
        })

        print(f"✅ {name} Results → Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")

        # Save confusion matrix
        if save_results:
            print_confusion_matrix(y_test, y_pred, name)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    print("\n=== 📊 Summary of Classifier Performance ===")
    print(df_results)

    # Save results
    if save_results:
        df_results.to_csv("results/classifier_performance.csv", index=False)
        print("\n💾 Results saved to: results/classifier_performance.csv")

    # Visualization
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_results.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model', y='Score', hue='Metric', palette='viridis')
    plt.title("Classifier Performance Comparison (SmartFS + SmartHive)")
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_results:
        plt.savefig("results/classifier_comparison.png")
        plt.close()
    else:
        plt.show()

    return df_results


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    print("\n=== Evaluation Script: SmartFS + SmartHive ===")

    # Example: Load data prepared from SmartFS + SmartHive
    # Replace with your actual processed dataset paths
    try:
        X = np.load("results/X_selected.npy")
        y = np.load("results/y.npy")
    except:
        print("⚠️ Could not load dataset. Please ensure X_selected.npy and y.npy exist in /results.")
        exit()

    df_metrics = evaluate_models(X, y)
