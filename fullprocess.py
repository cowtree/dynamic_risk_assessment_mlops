

import json
import os
import logging
import subprocess
import numpy as np
import training
import ingestion
import scoring
import deployment
import diagnostics
import reporting

# logging
logging.basicConfig(filename='fullprocess.log', \
                level=logging.DEBUG, \
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# first, read ingestedfiles.txt
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
source_data_path = os.path.join(config['input_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
ingestedfiles = os.path.join(config['ingested_file'])
latestscore = os.path.join(config['latestscore_file'])
model_name = os.path.join(config['model_name'])

with open(os.path.join(dataset_csv_path,ingestedfiles)) as f:
    ingestedfiles = json.load(f)


def check_new_files():
    """Check for new files in the source data folder
    and merge them into one dataframe.
    """
    ingestion.merge_multiple_dataframe()
    ingestion.record_ingestion_datasets()


def run_full_process():
    """
    Run the full process of data ingestion,
    training, scoring, deployment,
    diagnostics, and reporting.
    """

    logging.info('Starting the full process...')

    # get the filenames from the source data folder, only csv files
    source_data_list = os.listdir(source_data_path)
    source_data_list = [x for x in source_data_list if x.endswith('.csv')]

    if source_data_list == []:
        print('There is no new data. Ending the process here.')
        exit()
    elif len(source_data_list) > 1:
        # compare source data with ingestedfiles
        ingested_list = ingestedfiles['filenames']

    if set(ingested_list) != set(source_data_list):
        logging.info('There is new data. Proceeding.')
        logging.info("No. of ingested files: %s", len(ingested_list))
        logging.info("No. of source data files: %s", len(source_data_list))
        # get new files
        check_new_files()

        logging.info("Retraining the model with new data...")
        training.train_model()
        recent_f1 = scoring.score_model()
        logging.info("Score of the new model: %s", recent_f1)

        # read the previous scores from the deployment directory
        with open(os.path.join(prod_deployment_path, latestscore), 'r') as f:
            previous_f1 = float(f.read())

        if (recent_f1 != previous_f1):
            logging.info('Model drift is observed. Proceeding the process.')

            if recent_f1 > np.min(previous_f1):
                logging.info('New model performs better than previous model.')
                logging.info('Re-deploying model to production...')
                deployment.store_model_into_pickle(model_name)

            # Generate diagnostics
            logging.info('Generating diagnostics...')
            diagnostics.dataframe_summary()
            diagnostics.execution_time()
            diagnostics.check_missing_values()
            diagnostics.outdated_packages_list()

            # Generate reports
            logging.info('Generating reports...')
            reporting.score_model()

            # run API calls
            logging.info('Running API calls...')
            subprocess.call(['python', 'apicalls.py'])

            logging.info('Process completed. Model re-trained and re-deployed.')

    else:
        logging.info('There is no new data. Ending process.')
        exit()


if __name__ == '__main__':
    run_full_process()
