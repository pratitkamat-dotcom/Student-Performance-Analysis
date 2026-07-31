import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path definitions
DATA_DIR = r"C:\student -performance-analyzer\Student-Performance-Analysis\dataset"
MODEL_DIR = r"C:\student -performance-analyzer\Student-Performance-Analysis\model"
SUMMARY_PATH = r"C:\student -performance-analyzer\Student-Performance-Analysis\notebooks\training_summary.txt"

# Load training and test splits
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train_raw = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
y_test_raw = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

# Convert G3 numeric grades into Binary Labels (Classification: Pass=1 if G3 >= 10, Else Fail=0)
y_train = (y_train_raw["G3"] >= 10).astype(int)
y_test = (y_test_raw["G3"] >= 10).astype(int)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

results = {}

with open(SUMMARY_PATH, "w", encoding="utf-8") as summary:
    summary.write("=== MODEL TRAINING & COMPARISON REPORT ===\n\n")
    summary.write(f"Dataset split details for Classification:\n")
    summary.write(f"* Train instances: {len(y_train)} ({y_train.sum()} Passes, {len(y_train) - y_train.sum()} Fails)\n")
    summary.write(f"* Test instances:  {len(y_test)} ({y_test.sum()} Passes, {len(y_test) - y_test.sum()} Fails)\n\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "model_object": model
        }
        
        summary.write(f"[{name}] Evaluation Metrics:\n")
        summary.write(f"   * Accuracy:  {accuracy:.4f}\n")
        summary.write(f"   * Precision: {precision:.4f}\n")
        summary.write(f"   * Recall:    {recall:.4f}\n")
        summary.write(f"   * F1-Score:  {f1:.4f}\n\n")

    # Recommend the best model based on Accuracy
    best_model_name = max(results, key=lambda k: results[k]["Accuracy"])
    best_model_data = results[best_model_name]

    summary.write("==============================================\n")
    summary.write(f"🏆 RECOMMENDED MODEL: {best_model_name}\n")
    summary.write(f"Accuracy: {best_model_data['Accuracy']:.4f}\n")
    summary.write("==============================================\n")

# Save the best model using JOBLIB!
best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
joblib.dump(best_model_data["model_object"], best_model_path)

print("Best model and report successfully saved using joblib!")
