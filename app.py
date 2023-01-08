import json
import os
import utils
from flask import Flask, jsonify
from dotenv import load_dotenv
from scoring import score_model
from diagnostics import model_predictions, \
    dataframe_summary, \
    execution_time,  \
    check_missing_values, \
    outdated_packages_list

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_name = os.path.join(config['model_name'])

prediction_model = None

@app.before_first_request
def load_model():
    """Load the model at the start of the app.
    """
    global prediction_model
    prediction_model = utils.load_ml_model(model_name)

@app.route("/prediction", methods=['POST', 'OPTIONS'])
def prediction(testdata_path: str = test_data_path):
    """Compute predictions for the dataset.
    """
    print(type(prediction_model))
    preds = model_predictions(prediction_model, testdata_path)
    return jsonify({'predictions': preds, 'status_code': 200})

@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    """Compute F1 score for the model.

    Returns:
        json: F1 score for the model and status code.
    """
    score = score_model(model_name)
    # add return value (a single F1 score number)
    return jsonify({'f1score': score, 'status_code': 200})

@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    """Compute summary statistics for the dataset.
    Returns:
        json: Mean/Median/Standard deviation for each
                column in the dataset nd status code.
    """
    summary_stats = dataframe_summary(dataset_csv_path)
    return jsonify({'summary_stats': summary_stats,
                    'status_code': 200})

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    """Compute diagnostics for the dataset.
    Returns:
        json: Execution time, Missing values, package list and status code.
    """

    # check timing and percent NA values
    exec_time = execution_time()
    missing_vals = check_missing_values(dataset_csv_path)
    list_packages = outdated_packages_list()

    return jsonify({'exec_time': exec_time,
                    'missing_vals (in %)': missing_vals,
                    'dependency_check': list_packages,
                    'status_code': 200})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
