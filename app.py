from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "student_performance_analyser_db_secret_key"

# Path definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "students.db")

# Load the trained machine learning model once at startup
try:
    print(f"Loading classifier model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading the model at startup: {e}")

# The 15 exact feature names in the exact columns order the model trained on
FEATURES = [
    'sex', 'age', 'address', 'Medu', 'Fedu', 
    'studytime', 'failures', 'schoolsup', 'famsup', 
    'romantic', 'goout', 'health', 'absences', 
    'G1', 'G2'
]

def init_db():
    """Create the SQLite database and students table if they do not exist."""
    print("Initializing student database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We store all inputs + ML output predictions + probability metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            roll_number TEXT,
            sex INTEGER,
            age INTEGER,
            address INTEGER,
            Medu INTEGER,
            Fedu INTEGER,
            studytime INTEGER,
            failures INTEGER,
            schoolsup INTEGER,
            famsup INTEGER,
            romantic INTEGER,
            goout INTEGER,
            health INTEGER,
            absences INTEGER,
            G1 INTEGER,
            G2 INTEGER,
            prediction TEXT,
            predicted_performance TEXT,
            pass_probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Run dynamic schema updates if table exists
    cursor.execute("PRAGMA table_info(students)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'student_name' not in columns:
        cursor.execute("ALTER TABLE students ADD COLUMN student_name TEXT")
    if 'roll_number' not in columns:
        cursor.execute("ALTER TABLE students ADD COLUMN roll_number TEXT")
    if 'predicted_performance' not in columns:
        cursor.execute("ALTER TABLE students ADD COLUMN predicted_performance TEXT")
        # Backfill values from existing prediction column
        cursor.execute("UPDATE students SET predicted_performance = prediction")
            
    conn.commit()
    conn.close()
    print("Database check completed successfully!")

# Initialize database
init_db()

@app.route('/')
def index():
    """Home landing page displaying general dashboard metrics from the SQLite DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Calculate dashboard statistics
        cursor.execute("SELECT COUNT(*) FROM students")
        total_evaluations = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM students WHERE prediction = 'PASS'")
        total_passes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM students WHERE prediction = 'FAIL'")
        total_fails = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(absences) FROM students")
        avg_absences = cursor.fetchone()[0]
        avg_absences = round(avg_absences, 1) if avg_absences else 0
        
        conn.close()
    except Exception as e:
        total_evaluations = 0
        total_passes = 0
        total_fails = 0
        avg_absences = 0
        print(f"Error loading dashboard stats: {e}")
        
    dashboard_stats = {
        'total': total_evaluations,
        'passes': total_passes,
        'fails': total_fails,
        'avg_absences': avg_absences
    }
    
    return render_template('index.html', stats=dashboard_stats)

@app.route('/predict_form')
def predict_form():
    """Render the Student Prediction Form page."""
    return render_template('predict_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Extract inputs, validate, save to SQLite, runs prediction, update SQLite, and render result."""
    if model is None:
        flash("Model Error: The ML classifier could not be loaded.", "danger")
        return redirect(url_for('predict_form'))
        
    try:
        # Extract inputs from form variables
        student_name = request.form.get('student_name', '').strip()
        roll_number = request.form.get('roll_number', '').strip()
        
        sex = int(request.form.get('sex', 0))
        age = int(request.form.get('age', 16))
        address = int(request.form.get('address', 1))
        Medu = int(request.form.get('Medu', 2))
        Fedu = int(request.form.get('Fedu', 2))
        studytime = int(request.form.get('studytime', 2))
        failures = int(request.form.get('failures', 0))
        schoolsup = int(request.form.get('schoolsup', 0))
        famsup = int(request.form.get('famsup', 0))
        romantic = int(request.form.get('romantic', 0))
        goout = int(request.form.get('goout', 3))
        health = int(request.form.get('health', 3))
        absences = int(request.form.get('absences', 0))
        G1 = int(request.form.get('G1', 10))
        G2 = int(request.form.get('G2', 10))
        
        # Validations
        if not student_name or not roll_number:
            flash("Validation Error: Student Name and Roll Number are required.", "warning")
            return redirect(url_for('predict_form'))
        if not (15 <= age <= 22):
            flash("Validation Error: Age must be between 15 and 22.", "warning")
            return redirect(url_for('predict_form'))
        if not (0 <= absences <= 93):
            flash("Validation Error: Absences must be between 0 and 93.", "warning")
            return redirect(url_for('predict_form'))
        if not (0 <= G1 <= 20) or not (0 <= G2 <= 20):
            flash("Validation Error: Term grades G1 and G2 must be between 0 and 20.", "warning")
            return redirect(url_for('predict_form'))
            
        # Reconstruct DataFrame with exact column order
        student_features = pd.DataFrame([{
            'sex': sex, 'age': age, 'address': address, 'Medu': Medu, 'Fedu': Fedu,
            'studytime': studytime, 'failures': failures, 'schoolsup': schoolsup, 
            'famsup': famsup, 'romantic': romantic, 'goout': goout, 'health': health,
            'absences': absences, 'G1': G1, 'G2': G2
        }])
        
        # Perform machine learning prediction
        prediction_val = model.predict(student_features)[0]
        probabilities = model.predict_proba(student_features)[0]
        
        prediction_text = 'PASS' if prediction_val == 1 else 'FAIL'
        pass_prob = round(probabilities[1] * 100, 2)
        fail_prob = round(probabilities[0] * 100, 2)
        
        # Save Student record + prediction results into SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO students (
                student_name, roll_number, sex, age, address, Medu, Fedu, studytime, failures, schoolsup, 
                famsup, romantic, goout, health, absences, G1, G2, prediction, predicted_performance, pass_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_name, roll_number, sex, age, address, Medu, Fedu, studytime, failures, schoolsup,
            famsup, romantic, goout, health, absences, G1, G2, prediction_text, prediction_text, pass_prob
        ))
        conn.commit()
        conn.close()
        
        # Get the creation time
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Structure variables to render result page
        result_details = {
            'student_name': student_name,
            'roll_number': roll_number,
            'prediction': prediction_text,
            'pass_prob': pass_prob,
            'fail_prob': fail_prob,
            'created_at': current_time,
            'features': {
                'sex': sex, 'age': age, 'address': address, 'Medu': Medu, 'Fedu': Fedu,
                'studytime': studytime, 'failures': failures, 'schoolsup': schoolsup,
                'famsup': famsup, 'romantic': romantic, 'goout': goout, 'health': health,
                'absences': absences, 'G1': G1, 'G2': G2
            }
        }
        
        return render_template('result.html', result=result_details)
        
    except ValueError:
        flash("Input Format Error: Ensure all fields are filled with correct numeric formats.", "warning")
        return redirect(url_for('predict_form'))
    except Exception as e:
        flash(f"SQL/Model Operation Error: {str(e)}", "danger")
        return redirect(url_for('predict_form'))

@app.route('/history')
def history():
    """Fetch all history records from sqlite for displaying in table template."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Using sqlite3.Row configuration to access values by keys
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students ORDER BY created_at DESC")
        history_records = cursor.fetchall()
        conn.close()
    except Exception as e:
        history_records = []
        flash(f"Error querying history: {e}", "danger")
        
    return render_template('history.html', records=history_records)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear all records from database history."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students")
        conn.commit()
        conn.close()
        flash("History cleared successfully!", "success")
    except Exception as e:
        flash(f"Error clearing logs: {e}", "danger")
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)