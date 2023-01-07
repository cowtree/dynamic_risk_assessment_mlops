
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deploy_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(data_path : str = test_data_path):
    """Model predictions using the deployed model

    Args:
        data_path (str): Path to the test data

    Returns:
        list: Predictions
    """
    #read the deployed model and a test dataset, calculate predictions

    model = pd.read_pickle(os.path.join(prod_deploy_path,'trainedmodel.pkl'))
    testdata=pd.read_csv(data_path + '/testdata.csv')
    x_test = testdata[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    predicted = model.predict(x_test)

    return predicted.tolist()

##################Function to get summary statistics
def dataframe_summary(data_path : str = dataset_csv_path):
    """This function will calculate summary statistics for finaldata.csv

    Args:
        data_path (str, optional): Path to the dataset. Defaults to dataset_csv_path.

    Returns:
        dict: Summary statistics
    """    
    #calculate summary statistics here

    # calculate mean, median, standard deviations for all columns from finaldata.csv
    testdata=pd.read_csv(data_path + '/finaldata.csv')

    # get numeric columns
    num_cols = testdata.select_dtypes(include=np.number).columns

    # calculate mean, median, standard deviations for all numeric columns and append to a list
    mean = testdata[num_cols].mean().tolist()
    median = testdata[num_cols].median().tolist()
    std = testdata[num_cols].std().tolist()
    summary_stats = {'mean' : mean,
                    'median' : median,
                    "std" : std}

    return summary_stats

##################Function to get timings
def execution_time():
    """This function will calculate the time taken to run ingestion.py and training.py

    Returns:
        list: Time taken to run ingestion.py and training.py
    """
    #calculate timing of training.py and ingestion.py

    # calculate time taken to run ingestion.py
    start_time = timeit.default_timer()
    subprocess.call(['python', 'ingestion.py'])
    ingestion_time = timeit.default_timer() - start_time
    
    # calculate time taken to run training.py
    start_time = timeit.default_timer()
    subprocess.call(['python', 'training.py'])
    training_time = timeit.default_timer() - start_time

    return [ingestion_time, training_time]


######## Check for missing values
def check_missing_values(data_path : str = dataset_csv_path):
    """This function will check for missing values in finaldata.csv

    Args:
        data_path (str, optional): Path to the dataset . Defaults to dataset_csv_path.

    Returns:
        float: Percentage of missing values
    """    
    #check for missing values in finaldata.csv

    # read finaldata.csv
    testdata=pd.read_csv(data_path + '/finaldata.csv')

    # check for missing values
    missing_values = testdata.isnull().sum().sum()

    # calculate percentage of missing values
    total_cells = np.product(testdata.shape)
    missing_values_percent = (missing_values/total_cells) * 100

    return missing_values_percent


##################Function to check dependencies
def outdated_packages_list():
    """This function will check for outdated packages and return a list of installed packages, outdated packages and requirements

    Returns:
        dict: Dictionary containing installed packages, outdated packages and requirements
    """

    # get all packages installed in the environment
    installed = subprocess.check_output(['pip', 'list'])
    # get packages that are outdated
    broken = subprocess.check_output(['pip', 'list', '--outdated'])
    # requirements
    requirements = subprocess.check_output(['pip', 'freeze'])
    # get version of scikit-learn
    sklearninfo=subprocess.check_output(['pip', 'show', 'scikit-learn'])

    return {
        'installed_packages' : installed.decode('utf-8'),
        'outdated_packages' : broken.decode('utf-8'),
        'requirements' : requirements.decode('utf-8'),
        'scikit-learn_version' : sklearninfo.decode('utf-8')
    }

if __name__ == '__main__':
    model_predictions(dataset_csv_path)
    dataframe_summary()
    execution_time()
    check_missing_values()
    outdated_packages_list()





    
