from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl nahi mila. Pehle train_model.py run karein.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        feature_order = [
            'Age', 'Gender', 'Water_Intake_Liters', 'Urination_Frequency', 
            'Physical_Activity_Level', 'Sitting_Hours', 'Salt_Intake_Level', 
            'Protein_Intake', 'Sleep_Hours', 'Smoking', 'Alcohol', 'BP_Level', 'Diabetes'
        ]
        
        input_features = [float(data[col]) for col in feature_order]
        
        # Prediction
        prediction = model.predict([input_features])[0]
        
        result = "Risk of kidney disease is there(kidney test is recommended))" if prediction == 1 else "Low Risk (Healthy)"
        return jsonify({'prediction': result, 'status': 'success'})
    
    except Exception as e:
        return jsonify({'prediction': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)