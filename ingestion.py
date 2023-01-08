import os
import json
from datetime import datetime
import pandas as pd
import glob

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
concat_file = config['concatfile']

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


def save_clean_data(df, filename):
    """This function will save the clean data
    """
    df.to_csv(os.path.join(output_folder_path, filename), index=False)

def merge_multiple_dataframe():
    """This function will merge multiple dataframes into one dataframe
    """
    # check for datasets, compile them together, and write to an output file
    datasets = glob.glob(f'{input_folder_path}/*.csv')
    df_list = pd.concat(map(pd.read_csv, datasets))
    clean_df = df_list.drop_duplicates()
    save_clean_data(clean_df, concat_file)

def record_ingestion_datasets():
    """This function will record the ingestion datasets
    """
    datatime_obj = datetime.now()
    thetimenow = str(datatime_obj.year) + '/' + \
        str(datatime_obj.month) + '/' + str(datatime_obj.day)

    allrecords = {
        'input_folder_path': [],
        'filenames': [],
        'length_dataset': [],
        'thetimenow': []
    }

    # record the ingested datasets
    for directory in [input_folder_path]:
        filenames = os.listdir(os.path.join(os.getcwd(), directory))

        for each_filename in filenames:
            if each_filename.endswith('.csv'):
                df1 = pd.read_csv(
                    os.path.join(
                        os.getcwd(),
                        directory,
                        each_filename))

                allrecords['input_folder_path'].append(input_folder_path)
                allrecords['filenames'].append(each_filename)
                allrecords['length_dataset'].append(len(df1.index))
                allrecords['thetimenow'].append(thetimenow)

    ingested_file = os.path.join(output_folder_path, "ingestedfiles.txt")
    # Serializing json
    json_object = json.dumps(allrecords)

    # Writing to sample.json
    with open(ingested_file, "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    merge_multiple_dataframe()
    record_ingestion_datasets()
