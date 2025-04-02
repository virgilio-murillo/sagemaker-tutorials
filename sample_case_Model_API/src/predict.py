
import pickle  # Replace joblib with pickle
import os
import pandas as pd
from io import StringIO
import json


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

def model_fn(model_dir):
    # Print the contents of /opt/ml in a tree-like structure
    print("Contents of /opt/ml:")
    print_directory_tree('/opt/ml')

    # Print the current working directory (.)
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)
    print_directory_tree(current_directory)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Print the directory of the script (folder/subfolder)
    print("Script directory:", script_directory)
    print_directory_tree(script_directory)

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
