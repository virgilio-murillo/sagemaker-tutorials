# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Sagemaker Tutorial Series

# %% [markdown]
# ### Tutorial - 1 Mobile Price Classification using SKLearn Custom Script in Sagemaker

# %% [markdown]
# Data Source - https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?resource=download

# %% [markdown]
# ### Let's divide the workload
# 1. Initialize Boto3 SDK and create S3 bucket. 
# 2. Upload data in Sagemaker Local Storage. 
# 3. Data Exploration and Understanding.
# 4. Split the data into Train/Test CSV File. 
# 5. Upload data into the S3 Bucket.
# 6. Create Training Script
# 7. Train script in-side Sagemaker container. 
# 8. Store Model Artifacts(model.tar.gz) into the S3 Bucket. 
# 9. Deploy Sagemaker Endpoint(API) for trained model, and test it. 

# %%
import sklearn # Check Sklearn version
sklearn.__version__

# %%
# !python --version

# %% [markdown]
# ## 1. Initialize Boto3 SDK and create S3 bucket. 

# %%
import numpy as np
from sagemaker import get_execution_role
import sagemaker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import datetime
import time
import tarfile
import boto3
import pandas as pd

sm_boto3 = boto3.client("sagemaker")
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = 'mainbucketrockhight5461' # Mention the created S3 bucket name here
print("Using bucket " + bucket)
# hi
print(f"sagemaker version: {sagemaker.__version__}")

# %% [markdown]
# ## 3. Data Exploration and Understanding.

# %%
df = pd.read_csv("mob_price_classification_train.csv")

# %%
df.head()

# %%
df.shape

# %%
# ['Low_Risk','High_Risk'],[0,1]
df['price_range'].value_counts(normalize=True)

# %%
df.columns

# %%
df.shape

# %%
# Find the Percentage of Values are missing
df.isnull().mean() * 100

# %%
features = list(df.columns)
features

# %%
label = features.pop(-1)
label

# %%
x = df[features]
y = df[label]

# %%
x.head()

# %%
# {0: 'Low_Risk',1: 'High_Risk'}
y.head()

# %%
x.shape

# %%
y.value_counts()

# %%
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=0)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %% [markdown]
# ## 4. Split the data into Train/Test CSV File. 

# %%
trainX = pd.DataFrame(X_train)
trainX[label] = y_train

testX = pd.DataFrame(X_test)
testX[label] = y_test

# %%
print(trainX.shape)
print(testX.shape)

# %%
trainX.head()

# %%
trainX.isnull().sum()

# %%
testX.isnull().sum()

# %% [markdown]
# ## 5. Upload data into the S3 Bucket.

# %%
trainX.to_csv("train-V-1.csv",index = False)
testX.to_csv("test-V-1.csv", index = False)

# %%
# send data to S3. SageMaker will take training data from s3
sk_prefix = "sagemaker/mobile_price_classification/sklearncontainer"
trainpath = sess.upload_data(
    path="train-V-1.csv", bucket=bucket, key_prefix=sk_prefix
)

testpath = sess.upload_data(
    path="test-V-1.csv", bucket=bucket, key_prefix=sk_prefix
)

# %%
testpath

# %%
trainpath

# %% [markdown]
# ## 6. Create Training Script

# %%
# %%writefile train.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sklearn
import pickle  # Replace joblib with pickle
import argparse
import os
import pandas as pd

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()
    
    print("SKLearn Version: ", sklearn.__version__)
    # Removed Joblib version print since joblib is no longer used

    print("[INFO] Reading data")
    print()
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
    features = list(train_df.columns)
    label = features.pop(-1)
    
    print("Building training and testing datasets")
    print()
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print('Column order: ')
    print(features)
    print()
    
    print("Label column is: ", label)
    print()
    
    print("Data Shape: ")
    print()
    print("---- SHAPE OF TRAINING DATA (85%) ----")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("---- SHAPE OF TESTING DATA (15%) ----")
    print(X_test.shape)
    print(y_test.shape)
    print()
    
    print("Training RandomForest Model.....")
    print()
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose=3, n_jobs=-1)
    model.fit(X_train, y_train)
    print()
    
    # Change file name to model.pkl and use pickle to save
    model_path = os.path.join(args.model_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model persisted at " + model_path)
    print()

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print()
    print("---- METRICS RESULTS FOR TESTING DATA ----")
    print()
    print("Total Rows are: ", X_test.shape[0])
    print('[TESTING] Model Accuracy is: ', test_acc)
    print('[TESTING] Testing Report: ')
    print(test_rep)

# %%
# %%writefile predict.py

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


# %%
from predict import model_fn

# %%
# Test model_fn
print("Testing model_fn:")
loaded_model = model_fn('.')
print("Model loaded successfully")

# %%
# ! python train.py --n_estimators 100 \
#                    --random_state 0 \
#                    --model-dir ./ \
#                    --train ./ \
#                    --test ./ \

# %% [markdown]
# ## 7. save the model.pkl into model.tar.gz

# %%
import tarfile

with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model.pkl")

# %% [markdown]
# ### 7.1 we will test the model locally

# %%
import pickle
import pandas as pd
from io import StringIO
import numpy as np

# Assuming testX and features are already defined from your earlier code
# If not, load your test data here, e.g.:
# test_df = pd.read_csv("test-V-1.csv")
# features = list(test_df.columns[:-1])  # Adjust based on your data
# testX = test_df[features]

# Step 1: Load the model from model.pkl
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Step 2: Prepare test data (similar to the endpoint)
# Take the first two rows of testX[features]
test_data = testX[features][0:2].values.tolist()

# Convert to CSV string (mimicking CSVSerializer)
csv_buffer = StringIO()
pd.DataFrame(test_data).to_csv(csv_buffer, header=False, index=False)
csv_data = csv_buffer.getvalue()

# Step 3: Parse the CSV string back to a DataFrame (mimicking input_fn in predict.py)
df_input = pd.read_csv(StringIO(csv_data), header=None)

# Step 4: Make predictions using the loaded model
predictions = model.predict(df_input)

# Step 5: Convert predictions to NumPy array (mimicking NumpyDeserializer)
result = np.array(predictions)

# Step 6: Print the result
print("Predictions:", result)

# %% [markdown]
# ## 8. Store Model Artifacts(model.tar.gz) into the S3 Bucket. 

# %%
s3 = boto3.client('s3')

# Upload the tar.gz file to S3
s3.upload_file("model.tar.gz", bucket, "models/model.tar.gz")
model_data = f"s3://{bucket}/models/model.tar.gz"

print(f"model data: {model_data}")

# %%
ecr_image = 

# %% [markdown]
# ## 9. Deploy Sagemaker Endpoint(API) for trained model, and test it. 

# %%
from sagemaker.model import Model
from time import gmtime, strftime

model_name = "Custom-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(f"framework version: {sklearn.__version__}")
model = Model(
    name =  model_name,
    model_data=model_data,
    role=get_execution_role(),
    entry_point="predict.py",
)

# %%
endpoint_name = "Custom-sklearn-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("EndpointName={}".format(endpoint_name))

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=endpoint_name,
)

# %%
testX[features][0:2].values.tolist()

# %%
import pandas as pd
from io import StringIO
from sagemaker.deserializers import NumpyDeserializer
from sagemaker.serializers import CSVSerializer

# Convert testX[features][0:2] to CSV string
test_data = testX[features][0:2].values.tolist()
csv_buffer = StringIO()
pd.DataFrame(test_data).to_csv(csv_buffer, header=False, index=False)
csv_data = csv_buffer.getvalue()

# Set up the predictor with appropriate serializer and deserializer
predictor.serializer = CSVSerializer()
predictor.deserializer = NumpyDeserializer()

# Use predictor.predict with explicit content type
predictor.content_type = "text/csv"  # Set the content type for the request
predictor.accept = "application/x-npy"  # Set the accept type for the response

# Make the prediction
result = predictor.predict(csv_data)
print(result)

# %% [markdown]
# ## Don't forget to delete the endpoint !

# %%
sm_boto3.delete_endpoint(EndpointName=endpoint_name)

# %% [markdown]
# ### Don't forget to Subscribe Machine Learning Hub YouTube Channel. 

# %%
