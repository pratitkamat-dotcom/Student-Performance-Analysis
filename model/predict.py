# ==========================================
# predict.py - Simple ML Prediction Script
# ==========================================
# This script loads our trained machine learning model and uses it 
# to predict whether a sample student will Pass or Fail their class.

import pandas as pd
import joblib

# 1. PATH TO THE TRAINED MODEL
MODEL_PATH = "model/best_model.pkl"

# 2. LOAD THE SAVED MODEL
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!\n")

# 3. SAMPLE STUDENT INPUT
# We create a dictionary containing features for one hypothetical student.
sample_student = {
    'sex': [1],
    'age': [17],
    'address': [1],
    'Medu': [4],
    'Fedu': [3],
    'studytime': [3],
    'failures': [0],
    'schoolsup': [0],
    'famsup': [1],
    'romantic': [0],
    'goout': [2],
    'health': [4],
    'absences': [2],
    'G1': [12],
    'G2': [13]
}

# 4. CONVERT TO PANDAS DATAFRAME
# Scikit-learn models expect input as a structured table (DataFrame) 
# where column names exactly match the features it was trained on.
student_df = pd.DataFrame(sample_student)

print("Calculating predictions for sample student attributes:")
print(student_df.to_string(index=False))
print("-" * 50)

# 5. MAKE PREDICTION
prediction = model.predict(student_df)
probabilities = model.predict_proba(student_df)

prediction_class = prediction[0]
pass_probability = probabilities[0][1] * 100
fail_probability = probabilities[0][0] * 100

# 6. DISPLAY RESULTS
if prediction_class == 1:
    print("[SUCCESS] Prediction: The student is predicted to PASS the class!")
    print(f"Confidence: {pass_probability:.2f}% chance of Passing.")
else:
    print("[WARNING] Prediction: The student is predicted to FAIL the class.")
    print(f"Confidence: {fail_probability:.2f}% chance of Failing.")
print("-" * 50)
