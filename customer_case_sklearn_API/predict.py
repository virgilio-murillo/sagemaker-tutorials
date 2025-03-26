
import pickle
import os
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    """Load the trained model.pkl"""
    with open(os.path.join(model_dir, "model.pkl"), 'rb') as f:
        return pickle.load(f)

def input_fn(request_body, request_content_type):
    """Parse CSV input into DataFrame with correct columns"""
    if request_content_type == "text/csv":
        # Hardcoded columns to match your model's requirements
        columns = [
            'timestamp',
            'in_data',
            'decline_v2a_debit',
            'days_since_sms_otp_success',
            'days_since_receiver_first_seen',
            'days_since_device_first_seen',
            'dda_age_in_days'
        ]
        df = pd.read_csv(StringIO(request_body.strip()), header=None)
        df.columns = columns  # Assign correct column names
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Run prediction on preprocessed DataFrame"""
    return model.predict(input_data)
