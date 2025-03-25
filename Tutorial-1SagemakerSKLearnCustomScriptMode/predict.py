
import pickle  # Replace joblib with pickle
import os
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    # Load the model using pickle from model.pkl
    with open(os.path.join(model_dir, "model.pkl"), 'rb') as f:
        clf = pickle.load(f)
    return clf

def input_fn(request_body, request_content_type):
    print(request_body)
    print(request_content_type)
    if request_content_type == "text/csv":
        request_body = request_body.strip()
        try:
            df = pd.read_csv(StringIO(request_body), header=None)
            return df
        except Exception as e:
            print(e)
    else:
        return "Please use Content-Type = 'text/csv' and, send the request!!"

def predict_fn(input_data, model):
    if isinstance(input_data, pd.DataFrame):
        prediction = model.predict(input_data)
        print(prediction)
        return prediction
    else:
        return input_data
