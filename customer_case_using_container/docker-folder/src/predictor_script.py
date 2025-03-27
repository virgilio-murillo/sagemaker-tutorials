# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback

import flask
import pandas as pd

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            print("Loading model from disk...")
            with open(os.path.join(model_path, "decision-tree-model.pkl"), "rb") as inp:
                cls.model = pickle.load(inp)
            print("Model successfully loaded")
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        print("Starting prediction process")
        clf = cls.get_model()
        predictions = clf.predict(input)
        print(f"Completed predictions for {len(predictions)} samples")
        return predictions


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    # Generate directory tree
    model_path = "/opt/ml/model/"
    tree_output = generate_directory_tree(model_path)
    print(f"tree output: {tree_output}")
    
    status = 200 if health else 404
    print(f"Health check {'passed' if health else 'failed'} with status {status}")
    return flask.Response(response="\n", status=status, mimetype="application/json")

def generate_directory_tree(startpath):
    """Generate a directory tree structure similar to the tree command"""
    tree = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree.append(f"{subindent}{f}")
    return '\n'.join(tree)
    
@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    print("\nReceived prediction request")
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        print("Processing CSV input")
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        # In predictor.py, modify the CSV reading line in the /invocations route
        data = pd.read_csv(s, header=0)  # Changed from header=None to header=0
        print(f"Loaded DataFrame with shape {data.shape}")
    else:
        print(f"Unsupported content type: {flask.request.content_type}")
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )

    print("Invoked with {} records".format(data.shape[0]))

    # Do the prediction
    print("Generating predictions")
    predictions = ScoringService.predict(data)
    print("Predictions formatted for response")

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    print("Returning prediction response")
    return flask.Response(response=result, status=200, mimetype="text/csv")