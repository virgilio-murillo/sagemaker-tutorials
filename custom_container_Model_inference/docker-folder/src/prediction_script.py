import pickle
import pandas as pd
from io import StringIO
import numpy as np
from flask import Flask, request, jsonify
import flask
import sklearn # Check Sklearn version
import xgboost

# Instantiate Flask app
app = Flask(__name__)
print("Flask app instantiated")  # Debug print

# Define the model path
MODEL_PATH = "/opt/ml/model/model.pkl"
print(f"Model path set to: {MODEL_PATH}")  # Debug print

# Load the model from the pickle file
try:
    with open(MODEL_PATH, "rb") as f:
        print("Attempting to load model from pickle file...")  # Debug print
        model = pickle.load(f)
        print("Model loaded successfully")  # Debug print
        print(f"Model type: {type(model)}")  # Debug print
except Exception as e:
    print(f"Error loading model: {str(e)}")  # Debug print
    raise

# Health check endpoint
@app.route('/ping', methods=['GET'])
def ping():
    print("Received ping request")  # Debug print
    return "", 200

# Prediction endpoint
@app.route('/invocations', methods=['POST'])
def predict():
    print("Received prediction request")  # Debug print
    print(f"SKlearn version {sklearn.__version__}")
    print(f"xgboost version {xgboost.__version__}")
    try:
        # Get CSV data from the POST request
        data = request.get_data().decode('utf-8')
        print(f"Received data (first 200 chars): {data[:200]}")  # Debug print
        
        # Convert CSV string to a Pandas DataFrame
        df_input = pd.read_csv(StringIO(data), header=None)
        print("Successfully converted input to DataFrame")  # Debug print
        print(f"DataFrame shape: {df_input.shape}")  # Debug print
        print(f"First few rows:\n{df_input.head()}")  # Debug print
        
        # Make predictions using the loaded model
        print("Making predictions...")  # Debug print
        predictions = model.predict(df_input)
        print("Predictions completed")  # Debug print
        print(f"Predictions type: {type(predictions)}")  # Debug print
        print(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")  # Debug print
        print(f"First few predictions: {predictions[:5] if len(predictions) > 5 else predictions}")  # Debug print
        
        # Convert predictions to a list for JSON serialization
        result = predictions.tolist()
        print("Converted predictions to list")  # Debug print
        
        # Return predictions as JSON
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500