import json
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns

from diagnostics import model_predictions

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

##############Function for reporting
def score_model():
    """This function will score the model
    """    
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    # get predictions of deployed model in diagnostics.py running on the test data
    predicted = model_predictions()

    # load test data
    testdata=pd.read_csv(test_data_path + '/testdata.csv')
    # get actual values
    y = testdata['exited'].values.reshape(-1,1)
    
    # compute confusion matrix
    cnf_matrix = metrics.confusion_matrix(y, predicted)

    # plot confusion matrix
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

    # if confusion matrix png already exists, add counter to filename
    if os.path.exists(os.path.join(model_path,'confusionmatrix.png')):
        i = 2
        while os.path.exists(os.path.join(model_path,f'confusionmatrix{i}.png')):
            i += 1
        plt.savefig(os.path.join(model_path,f'confusionmatrix{i}.png'))
    else:
        plt.savefig(os.path.join(model_path,'confusionmatrix.png'))
    
if __name__ == '__main__':
    score_model()
