import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Path definitions
X_PATH = r"C:\student -performance-analyzer\Student-Performance-Analysis\dataset\X_processed.csv"
Y_PATH = r"C:\student -performance-analyzer\Student-Performance-Analysis\dataset\y_processed.csv"
DATASET_DIR = r"C:\student -performance-analyzer\Student-Performance-Analysis\dataset"

# Load the processed datasets
X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH)

# Perform 80/20 train/test split
# random_state=42 is set to ensure the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print out the shapes for verification
print(f"=== SPLIT DATAFRAME SHAPES ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# Save splits to dataset directory
X_train.to_csv(os.path.join(DATASET_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(DATASET_DIR, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(DATASET_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(DATASET_DIR, 'y_test.csv'), index=False)

print("\nSplit successfully completed and saved into 'dataset/' folder!")
