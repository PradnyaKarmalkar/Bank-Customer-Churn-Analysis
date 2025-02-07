# app.py

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('churn_predict_model')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()

    # Extract input features
    features = [
        data['CreditScore'],
        data['Age'],
        data['Tenure'],
        data['Balance'],
        data['NumOfProducts'],
        data['HasCrCard'],
        data['IsActiveMember'],
        data['EstimatedSalary'],
        data['Geography_Germany'],
        data['Geography_Spain'],
        data['Gender_Male']
    ]

    # Make prediction using the loaded model
    prediction = model.predict([features])[0]

    # Return prediction as JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
