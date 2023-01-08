import timeit
import os
import json
import subprocess
import pandas as pd

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = config['output_folder_path']
test_data_path = config['test_data_path']
prod_deploy_path = config['prod_deployment_path']
test_data_file = config['test_data_file']
concat_file = config['concatfile']

def model_predictions(model : object = None,
                        data_path: str = test_data_path):
    """Model predictions using the deployed model

    Args:
        data_path (str): Path to the test data

    Returns:b
        list: Predictions
    """
    # read the deployed model and a test dataset, calculate predictions
    testdata = pd.read_csv(os.path.join(data_path,test_data_file))
    testdata.drop(['corporation', 'exited'], inplace=True, axis=1)

    if model is None:
        predicted = model.predict(testdata) 
        return predicted.tolist()
    else:
        print('No model found')
        exit()

def dataframe_summary(data_path: str = dataset_csv_path):
    """This function will calculate summary statistics

    Args:
        data_path (str, optional): Path to the dataset. Defaults to dataset_csv_path.

    Returns:
        dict: Summary statistics
    """
    # calculate mean, median, standard deviations for all columns from
    testdata = pd.read_csv(os.path.join(data_path,concat_file))

    testdata.drop(['corporation', 'exited'], inplace=True, axis=1)
    result_summary = testdata.agg(["mean", "median", "std"]).to_numpy()

    # calculate mean, median, standard deviations for all numeric columns and
    # append to a list
    return  {'mean': result_summary[0,:].tolist(),
            'median': result_summary[1,:].tolist(),
            "std": result_summary[2,:].tolist()}


def execution_time(repeat_cnt: int = 1):
    """This function will calculate the time taken to run ingestion.py and training.py

    Args:
        repeat_cnt (int, optional): Number of times to repeat the timing. Defaults to 500.

    Returns:
        list: Time taken to run ingestion.py and training.py
    """
    # calculate timing of training.py and ingestion.py
    avg_ingestion_time = timeit.timeit(
        stmt="subprocess.call(['python', 'ingestion.py'])",
        setup="import subprocess",
        number=repeat_cnt)
    avg_training_time = timeit.timeit(
        stmt="subprocess.call(['python', 'training.py'])",
        setup="import subprocess",
        number=repeat_cnt)

    return [avg_ingestion_time, avg_training_time]


def check_missing_values(data_path: str = dataset_csv_path):
    """This function will check for missing values

    Args:
        data_path (str, optional): Path to the dataset . Defaults to dataset_csv_path.

    Returns:
        float: Percentage of missing values
    """

    testdata = pd.read_csv(os.path.join(data_path,concat_file))

    missing_values_df = testdata.isna().sum() / testdata.shape[0]
    return missing_values_df.values.tolist()


def outdated_packages_list():
    """
    This function will check for outdated packages and return a list of
    installed packages, outdated packages and requirements

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
    sklearninfo = subprocess.check_output(['pip', 'show', 'scikit-learn'])

    return {
        'installed_packages': installed.decode('utf-8'),
        'outdated_packages': broken.decode('utf-8'),
        'requirements': requirements.decode('utf-8'),
        'scikit-learn_version': sklearninfo.decode('utf-8')
    }


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    check_missing_values()
    outdated_packages_list()
