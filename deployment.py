from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import json


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

if not os.path.exists(prod_deployment_path):
    os.makedirs(prod_deployment_path)

####################function for deployment
def store_model_into_pickle(model):
    """Copy the latest pickled model, model score and ingested files into the deployment directory

    Args:
        model (str): model name
    """
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    os.system('cp ' + os.path.join(model_path,model) + ' ' + prod_deployment_path)
    os.system('cp ' + os.path.join(model_path,'latestscore.txt') + ' ' + prod_deployment_path)
    os.system('cp ' + os.path.join(dataset_csv_path,'ingestedfiles.txt') + ' ' + prod_deployment_path)

if __name__ == '__main__':
    store_model_into_pickle('trainedmodel.pkl')