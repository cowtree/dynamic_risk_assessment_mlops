import os
import json
import shutil
import logging



logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                              "%Y-%m-%d %H:%M:%S")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
latest_score_file = os.path.join(config['latestscore_file'])
ingested_file = os.path.join(config['ingested_file'])
model_name = os.path.join(config['model_name'])

if not os.path.exists(prod_deployment_path):
    os.makedirs(prod_deployment_path)

def store_model_into_pickle(model):
    """Copy the latest pickled model, model score and ingested files into the deployment directory

    Args:
        model (str): model name
    """
    shutil.copy2(os.path.join(model_path, model), prod_deployment_path)
    shutil.copy2(os.path.join(model_path, latest_score_file), prod_deployment_path)
    shutil.copy2(os.path.join(dataset_csv_path, ingested_file), prod_deployment_path)

if __name__ == '__main__':
    store_model_into_pickle(model_name)
