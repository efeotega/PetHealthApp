from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
from datetime import datetime
from database import get_db

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load model and feature columns
model_data = joblib.load('data/ai_model/pet_health_model.pkl')
model = model_data['model']
feature_columns = model_data['feature_columns']

def get_historical_data():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM health_logs ORDER BY timestamp DESC LIMIT 7')
    logs = cursor.fetchall()
    return logs

@app.route('/')
def dashboard():
    logs = get_historical_data()
    return render_template('dashboard.html', logs=logs)

@app.route('/log', methods=['POST'])
def log_entry():
    data = {
        'diet': request.form['diet'],
        'behavior': int(request.form['behavior']),
        'stool_appearance': request.form['stool_appearance'],
        'timestamp': datetime.now()
    }
    
    conn = get_db()
    conn.execute('''INSERT INTO health_logs 
                  (diet, behavior, stool_appearance, timestamp)
                  VALUES (?, ?, ?, ?)''',
                  (data['diet'], data['behavior'], 
                   data['stool_appearance'], data['timestamp']))
    conn.commit()
    
    return redirect(url_for('dashboard'))

def prepare_features(logs):
    # Convert logs to DataFrame
    df = pd.DataFrame(logs, columns=['id', 'diet', 'behavior', 'stool_appearance', 'timestamp'])
    
    # Create dummy variables
    df = pd.get_dummies(df, columns=['diet', 'stool_appearance'])
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            
    return df[feature_columns]

@app.route('/predict')
def predict():
    try:
        logs = get_historical_data()
        
        if len(logs) < 3:
            return render_template('results.html', 
                                 prediction="Insufficient data (need 3+ entries)",
                                 confidence="N/A")

        features = prepare_features(logs[-3:])
        predictions = model.predict(features)
        confidence = model.predict_proba(features).max(axis=1).mean()
        
        final_prediction = max(set(predictions), key=list(predictions).count)
        
        return render_template('results.html', 
                             prediction=final_prediction,
                             confidence=f"{confidence*100:.1f}%")
    except Exception as e:
        return render_template('results.html', 
                             prediction=f"Error: {str(e)}",
                             confidence="N/A")
def analyze_trends(data):
    # Create trend features
    avg_behavior = data['behavior'].mean()
    stool_counts = data['stool_appearance'].value_counts().to_dict()
    
    features = {
        'avg_behavior': avg_behavior,
        'most_common_stool': max(stool_counts, key=stool_counts.get),
        'diet_variety': len(data['diet'].unique())
    }
    
    return pd.DataFrame([features])

if __name__ == '__main__':
    app.run(debug=True)