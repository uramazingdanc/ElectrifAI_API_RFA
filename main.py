import pickle
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as model_file:
    rf_classifier = pickle.load(model_file)

# Load the saved preprocessors (LabelEncoder and MinMaxScaler)
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the saved column order used during training
with open('column_order.pkl', 'rb') as f:
    column_order = pickle.load(f)

# Custom function to handle unseen labels in LabelEncoder
def safe_label_encode(encoder, value, fallback=-1):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return fallback  # You can set a default or handle this case differently

# Preprocess the input data
def preprocess_data(data: dict):
    # Convert timestamp to datetime and extract features
    ts = pd.to_datetime(data['ts'], format='%Y%m%d%H%M%S')  # Convert the 'ts' string to a datetime object
    data['year'] = ts.year
    data['month'] = ts.month
    data['day'] = ts.day
    data['hour'] = ts.hour
    data['minute'] = ts.minute

    # Create a DataFrame for the incoming data
    df = pd.DataFrame([data])

    # Label encode and scale using the safe encoder for unseen labels
    df['meter_id'] = df['meter_id'].apply(lambda x: safe_label_encode(label_encoder, x))
    df['record_type'] = df['record_type'].apply(lambda x: safe_label_encode(label_encoder, x))

    # Scale the 'KWH' column
    df['KWH'] = scaler.transform(df[['KWH']])

    # Drop 'ts' after extracting the relevant features
    df = df.drop(columns=['ts'])

    # Reorder columns to match the original order used during training
    df = df[column_order]
    
    return df

# POST endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Preprocess incoming data
        input_data = preprocess_data(data)

        # Make prediction
        prediction = rf_classifier.predict(input_data)

        # Return the result
        result = "Anomaly" if prediction[0] == 1 else "Normal"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Optional: A simple root endpoint to test if the app is running
@app.route('/')
def index():
    return jsonify({"message": "Flask API is running!"})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
