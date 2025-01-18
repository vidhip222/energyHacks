from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder='../frontend', static_folder='../static')

# Load the model and scaler
model = joblib.load('xgboost_energy_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    required_fields = [
        'generation biomass', 'generation fossil brown coal/lignite', 
        'generation fossil gas', 'generation fossil hard coal', 
        'generation fossil oil', 'generation hydro pumped storage consumption', 
        'generation hydro run-of-river and poundage', 
        'generation hydro water reservoir', 'generation nuclear', 
        'generation other', 'generation other renewable', 
        'generation solar', 'generation waste', 'generation wind onshore', 
        'forecast solar day ahead', 'forecast wind onshore day ahead', 
        'total load forecast', 'total load actual', 'price day ahead'
    ]
    
    input_data = []
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
        input_data.append(float(data[field]))

    # Make prediction using your model
    prediction = model.predict([input_data])[0]
    
    # Determine action based on prediction
    current_price = float(data['price day ahead'])
    if prediction > current_price * 1.05:
        action = "Buy"
    elif prediction < current_price * 0.95:
        action = "Sell"
    else:
        action = "Store"
    
    return jsonify({
        'predicted_price': float(prediction),
        'action': action
    })

if __name__ == '__main__':
    app.run(debug=True)
