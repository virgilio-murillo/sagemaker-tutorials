
import os
import pandas as pd
from io import StringIO
import json
import pickle
from tornado.httputil import HTTPServerRequest

# Define the directory printing function from the original code
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
        print("Contents of /opt/ml:")
        print_directory_tree('/opt/ml')
        current_directory = os.getcwd()
        print("Current working directory:", current_directory)
        print_directory_tree(current_directory)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        print("Script directory:", script_directory)
        print_directory_tree(script_directory)
        # Load the model from /opt/ml/model/model.pkl, consistent with SageMaker-like environments
        with open('/opt/ml/model/model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def decode(self, request: HTTPServerRequest) -> str:
        # Decode the request body to a string, as in the template
        return request.body.decode("utf-8")

    def encode(self, response: dict) -> bytes:
        # Encode the response dictionary to JSON bytes, as in the template
        return json.dumps(response).encode("utf-8")

    def invoke(self, request: HTTPServerRequest) -> bytes:
        # Print directory structures every time invoke is called
        print("Contents of /opt/ml:")
        print_directory_tree('/opt/ml')
        current_directory = os.getcwd()
        print("Current working directory:", current_directory)
        print_directory_tree(current_directory)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        print("Script directory:", script_directory)
        print_directory_tree(script_directory)

        # Check content type, similar to input_fn in the original code
        if request.headers.get('Content-Type') != 'text/csv':
            return self.encode({"error": "Please use Content-Type = 'text/csv'"})

        # Decode and process the request body
        request_body = self.decode(request).strip()
        try:
            # Parse the CSV data into a DataFrame, as in input_fn
            df = pd.read_csv(StringIO(request_body), header=None)
            # Generate predictions using the model, as in predict_fn
            predictions = self.model.predict(df)
            # Convert predictions to a list for JSON serialization
            predictions_list = predictions.tolist()
            # Return predictions as a JSON response
            return self.encode({"predictions": predictions_list})
        except Exception:
            # Return an error if CSV parsing fails
            return self.encode({"error": "Invalid CSV data"})

# Instantiate the model
my_model = MyModel()

# Define the handler as per the template
async def handler(request: HTTPServerRequest):
    return my_model.invoke(request)
