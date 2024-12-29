from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('loan_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON request
    data = request.json
    df = pd.DataFrame([data])
    
    # Preprocess input data (use the same steps as during training)
    numerical_columns = [
        'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
        'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value'
    ]
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    
    # Prediction
    prediction = model.predict(df)[0]
    loan_status = "Approved" if prediction == 1 else "Rejected"
    return jsonify({'loan_status': loan_status})

if __name__ == '__main__':
    app.run(debug=True)
