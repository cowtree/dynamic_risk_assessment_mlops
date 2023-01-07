

import json
import os
import subprocess
import numpy as np
import training
import ingestion
import scoring
import deployment
import diagnostics
import reporting

##################Check and read new data

#first, read ingestedfiles.txt
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
source_data_path = os.path.join(config['input_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

with open(dataset_csv_path + '/ingestedfiles.txt') as f:
        ingestedfiles = json.load(f)

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

# get the filenames from the source data folder, only csv files

source_data_list = os.listdir(source_data_path)
source_data_list = [x for x in source_data_list if x.endswith('.csv')]


if source_data_list == []:
    print('There is no new data. Ending the process here.')
    exit()
elif len(source_data_list) > 1:
    # compare source data with ingestedfiles
    ingested_list = ingestedfiles['filenames']

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here

if set(ingested_list) != set(source_data_list):
    print('There is new data. Proceeding.')

    ##################Ingest new data
    ingestion.merge_multiple_dataframe()
    ingestion.record_ingestion_datasets()

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    #if the scores are different, then you should proceed. otherwise, do end the process here
    training.train_model()
    recent_f1 = scoring.score_model()
    
    # read the previous scores from the deployment directory
    with open(prod_deployment_path + '/latestscore.txt', 'r') as f:
        previous_f1 = float(f.read())

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    #if you found evidence for model drift, re-run the deployment.py script
    if (recent_f1 != previous_f1):
        print('There is model drift. Proceeding.')

        # if model score is better than previous model, then re-deploy the model to production
        '''
        #param test
        secondtest = recent_f1 < np.mean(previous_f1)-2*np.std(previous_f1)
        print(secondtest)

        #non-param test
        iqr = np.quantile(previous_f1, 0.75)-np.quantile(previous_f1, 0.25)
        thirdtest = recent_f1 < np.quantile(previous_f1, 0.25)-iqr*1.5
        print(thirdtest)
        '''

        if recent_f1 > np.min(previous_f1):
            print('New model performs better than previous model.')
            print('Re-deploying model to production...')
            deployment.store_model_into_pickle('trainedmodel.pkl')

        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model

        #Generate diagnostics
        print('Generating diagnostics...')
        diagnostics.dataframe_summary()
        diagnostics.execution_time()
        diagnostics.check_missing_values()
        diagnostics.outdated_packages_list()

        #Generate reports
        print('Generating reports...')
        reporting.score_model()

        # run API calls
        print('Running API calls...')
        subprocess.call(['python', 'apicalls.py'])

        print('Process completed. Model re-trained and re-deployed.')

else:
    print('There is no new data. Ending process.')
    exit()
