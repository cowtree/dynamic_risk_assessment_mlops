import os
import json
import pandas as pd
import sklearn.metrics as metrics
from utils import load_ml_model
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])
latestscore_file = os.path.join(config['latestscore_file'])
test_data_file = os.path.join(config['test_data_file'])
model_name = os.path.join(config['model_name'])

# check if model folder exists
if not os.path.exists(output_model_path):
    os.makedirs(output_model_path)


def score_model(model: object = None,
                test_data_filepath: str = test_data_file) -> float:
    """This function will score the model

    Args:
        model (str): Name of the model to be scored

    Returns:
        float: F1 score
    """

    testdata = pd.read_csv(test_data_filepath)
    testdata = testdata.drop(['corporation'], axis=1)
    y_train = testdata['exited'].values.reshape(-1, 1)
    predicted = model.predict(testdata.drop(['exited'], axis=1))

    f1score = metrics.f1_score(predicted, y_train)

    # write the f1 score to a file
    with open(os.path.join(output_model_path, latestscore_file), 'w') as file:
        file.write(str(f1score))

    return f1score

if __name__ == '__main__':
    score_model(load_ml_model(os.path.join(output_model_path, model_name)),
                os.path.join(test_data_path, test_data_file))
