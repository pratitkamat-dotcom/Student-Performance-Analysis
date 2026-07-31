import pandas as pd
import numpy as np
import os
import pickle

# Path definitions
DATASET_PATH = r"C:\student -performance-analyzer\Student-Performance-Analysis\dataset\student_data.csv"
OUTPUT_DIR = r"C:\student -performance-analyzer\Student-Performance-Analysis\dataset"

# Load the dataset
df = pd.read_csv(DATASET_PATH)
print(f"Original dataset shape: {df.shape}")

# 1. Handle Missing Values
# Double-check for missing values. If found, impute them.
missing_values = df.isnull().sum().sum()
if missing_values > 0:
    print(f"Warning: Found {missing_values} missing values. Handling them...")
    # Impute numerical with median, categorical with mode
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
else:
    print("1. No missing values detected in dataset.")

# 2. Remove Duplicate Rows
initial_count = len(df)
df = df.drop_duplicates()
final_count = len(df)
print(f"2. Duplicates status: Removed {initial_count - final_count} duplicate rows. Remaining rows: {final_count}")

# 3. Feature Selection & Categorical Encoding
# To create an interactive Flask UI, we select the 15 most important features that 
# capture social, educational, behavioral, and academic performance, avoiding 
# a massive form with 33 columns.
selected_features = [
    'sex', 'age', 'address', 'Medu', 'Fedu', 
    'studytime', 'failures', 'schoolsup', 'famsup', 
    'romantic', 'goout', 'health', 'absences', 
    'G1', 'G2'
]
target_col = 'G3'

print(f"3. Selecting {len(selected_features)} primary features for model training.")
df_selected = df[selected_features + [target_col]].copy()

# Encode Categorical Variables into Binary Numbers (0 and 1)
# Mapping yes/no, F/M, and Rural/Urban
binary_mappings = {
    'sex': {'F': 0, 'M': 1},
    'address': {'R': 0, 'U': 1},
    'schoolsup': {'no': 0, 'yes': 1},
    'famsup': {'no': 0, 'yes': 1},
    'romantic': {'no': 0, 'yes': 1}
}

print("4. Encoding categorical columns:")
for col, mapping in binary_mappings.items():
    if col in df_selected.columns:
        df_selected[col] = df_selected[col].map(mapping)
        print(f"   * Encoded '{col}' using mapping: {mapping}")

# Save the encoder mapping specifications to the model directory
encoder_path = r"C:\student -performance-analyzer\Student-Performance-Analysis\model\encoder_mappings.pkl"
os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
with open(encoder_path, 'wb') as f:
    pickle.dump(binary_mappings, f)
print(f"   * Encoder configurations saved to {encoder_path}")

# 5. Separate features (X) and target (y)
X = df_selected[selected_features]
y = df_selected[target_col]

print(f"5. Separated Features and Target:")
print(f"   * X shape: {X.shape}")
print(f"   * y shape: {y.shape}")

# Save the processed data for modeling phase
X.to_csv(os.path.join(OUTPUT_DIR, 'X_processed.csv'), index=False)
y.to_csv(os.path.join(OUTPUT_DIR, 'y_processed.csv'), index=False, header=[target_col])
print("Processed data files successfully saved to 'dataset/' directory.")
