from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from scoring import score_model
from diagnostics import model_predictions, \
                         dataframe_summary, \
                         execution_time,  \
                         check_missing_values, \
                         outdated_packages_list
import json
import os

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])

prediction_model = None

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def prediction(testdata_path : str = test_data_path):
    """Compute predictions for the dataset.
    """
    preds =   model_predictions(testdata_path)
    return jsonify({'predictions' : preds, 'status_code' : 200})
#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    """Compute F1 score for the model.

    Returns:
        json: F1 score for the model and status code.
    """    
    score = score_model('trainedmodel.pkl')
    return jsonify({'f1score' : score, 'status_code' : 200})  #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    """Compute summary statistics for the dataset.
    Returns:
        json: Mean/Median/Standard deviation for each column in the dataset nd status code.
    """
    summary_stats = dataframe_summary(dataset_csv_path)
    return jsonify({'summary_stats' : summary_stats, 'status_code'  : 200})

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """Compute diagnostics for the dataset.
    Returns:
        json: Execution time, Missing values, package list and status code.
    """

    #check timing and percent NA values
    exec_time = execution_time()
    missing_vals = check_missing_values(dataset_csv_path)
    list_packages = outdated_packages_list()

    return jsonify({'exec_time' : exec_time,
            'missing_vals (in %)' : missing_vals,
            'dependency_check' : list_packages,
            'status_code'  : 200}) #add return value for all diagnostics

if __name__ == "__main__":  
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)