# Student Performance Analysis

A Machine Learning mini-project built with Python, Flask, Pandas, NumPy, scikit-learn, and Bootstrap to analyze and predict student performance based on various academic and demographic features.

## Project Structure

```text
Student-Performance-Analysis/
│
├── dataset/             # Raw and processed datasets (e.g., student_data.csv)
├── notebooks/           # Jupyter Notebooks for Exploratory Data Analysis (EDA) and Model Training
├── model/               # Trained models, scalers, and encoder files (e.g., model.pkl)
├── templates/           # HTML templates for the Flask application (e.g., index.html)
├── static/              # Static assets serving the frontend
│   ├── css/            # Custom CSS stylesheets (e.g., style.css)
│   └── images/         # Images, logs, and illustrations for the UI
├── app.py               # Main Flask application file setting up server and prediction APIs
├── requirements.txt     # Python package dependencies
├── .gitignore           # Files and directories to be ignored by Git
└── README.md            # Project overview development guidelines
```

---

## Directory Descriptions

### 📂 `dataset/`
Contains the source dataset(s) (e.g. CSV, Excel files) containing student information such as study time, attendance, parental education level, and final grades. Both raw data and cleaned/processed datasets are tracked or referenced here.

### 📂 `notebooks/`
Contains Jupyter Notebooks (`.ipynb`) used during the development lifecycle:
* **Exploratory Data Analysis (EDA):** Visualization of relationships, distributions, and correlation matrices.
* **Feature Engineering & Selection:** Data cleaning, encoding categorical variables, handling outliers, etc.
* **Model Training & Hyperparameter Tuning:** Developing and evaluating models (e.g., Linear Regression, Decision Trees, Random Forests) and saving the final model object.

### 📂 `model/`
Contains serialized model files (e.g., `.pkl`, `.joblib`) after model training. This includes:
* Trained Scikit-Learn machine learning pipeline.
* Standard Scalers / MinMax Scalers used to normalize numerical features.
* Target encoders or one-hot encoders used to process categorical features.

### 📂 `templates/`
Holds the HTML view templates rendered by Flask via Jinja2:
* **`index.html`:** The primary dashboard interface where users can view insights and enter parameters to predict student performance.

### 📂 `static/`
Contains publicly accessible client-side assets:
* **`css/`:** CSS files defining the layout, custom typography, animations, glassmorphism, and color schemes.
* **`images/`:** Graphics, charts generated from notebooks (for display), and icons.

---

## Getting Started

### 1. Set Up Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask Web App

```bash
python app.py
```
After executing, open your browser and navigate to `http://127.0.0.1:5000/`.
