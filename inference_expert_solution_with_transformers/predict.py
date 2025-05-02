import os
import pandas as pd
from io import StringIO
import json
import pickle
import numpy as np
from tornado.httputil import HTTPServerRequest


def print_directory_tree(path, prefix="", is_last=True, ignore_venv=True):
    if os.path.isdir(path):
        dir_name = os.path.basename(path)
        if ignore_venv and dir_name in ['venv', 'env']:
            print(f"{prefix}{'└── ' if is_last else '├── '}{dir_name}/ (Python virtual environment, contents not listed)")
            return
        print(f"{prefix}{'└── ' if is_last else '├── '}{dir_name}/")
        new_prefix = prefix + ("    " if is_last else "│   ")
        contents = os.listdir(path)
        for i, item in enumerate(contents):
            is_last_item = i == len(contents) - 1
            item_path = os.path.join(path, item)
            print_directory_tree(item_path, new_prefix, is_last_item, ignore_venv)
    else:
        print(f"{prefix}{'└── ' if is_last else '├── '}{os.path.basename(path)}")
            
# Define the model class to fit the template
class MyModel:
    def __init__(self):
        # Load the model from /opt/ml/model/model.pkl, consistent with SageMaker-like environments
        with open('/opt/ml/model/model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        print("Model loaded successfully")


        
    def invoke(self, request: HTTPServerRequest) -> bytes:
        print("Contents of /opt/ml:")
        print_directory_tree('/opt/ml')
        current_directory = os.getcwd()
        print("Current working directory:", current_directory)
        print_directory_tree(current_directory)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        print("Script directory:", script_directory)
        print_directory_tree(script_directory)
        try:
            # Get and decode the request body
            body = request.body.decode('utf-8')
            print(f"Received request body: {body}")

            # Parse the JSON input
            input_data = json.loads(body)
            print(f"Parsed input: {input_data}")

            # Handle single object or array of objects
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                input_df = pd.DataFrame(input_data)
            else:
                print("Invalid input type: must be JSON object or array")
                return b'{"error": "Input must be a JSON object or array"}'

            print(f"Input DataFrame: {input_df}")

            # Make prediction using the loaded model
            prediction = self.model.predict(input_df)
            print(f"Prediction: {prediction}")
            print(f"Type of prediction: {type(prediction)}")
            if isinstance(prediction, dict):
                print(f"Keys in prediction: {list(prediction.keys())}")

            if isinstance(prediction, dict) and 'calibrated' in prediction:
                calibrated = prediction['calibrated']
                print(f"Calibrated prediction shape: {calibrated.shape}")
                if not isinstance(calibrated, np.ndarray):
                    print("Error: 'calibrated' is not a numpy array")
                    return b'{"error": "Calibrated prediction must be a numpy array"}'
                if len(input_df) == 1:
                    if calibrated.ndim == 1:
                        prediction_value = calibrated.tolist()
                    else:
                        prediction_value = calibrated[0].tolist()
                    response = {"prediction": prediction_value}
                else:
                    response = {"predictions": calibrated.tolist()}
            else:
                # Fallback for non-dictionary predictions
                print("Treating prediction as array")
                if len(input_df) == 1:
                    prediction_value = prediction[0]
                    if hasattr(prediction_value, 'item'):
                        prediction_value = prediction_value.item()
                    response = {"prediction": prediction_value}
                else:
                    response = {"predictions": prediction.tolist()}

            # Serialize response to JSON and encode to bytes
            response_bytes = json.dumps(response).encode('utf-8')
            print(f"Response: {response}")
            return response_bytes

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return b'{"error": "Invalid JSON"}'
        except Exception as e:
            print(f"Error during prediction: {e}")
            return b'{"error": "Internal server error"}'

# Instantiate the model
my_model = MyModel()

# Define the handler as per the template
async def handler(request: HTTPServerRequest):
    return my_model.invoke(request)