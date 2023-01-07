import pickle
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

# check if model folder exists
if not os.path.exists(output_model_path):
    os.makedirs(output_model_path)


#################Function for model scoring
def score_model(model = 'trainedmodel.pkl'):
    """This function will score the model

    Args:
        model (str): Name of the model to be scored

    Returns:
        float: F1 score
    """    

    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    with open(os.path.join(output_model_path,model), 'rb') as file:
        model = pickle.load(file)

    testdata=pd.read_csv(test_data_path + '/testdata.csv')

    x_train = testdata[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    y_train = testdata['exited'].values.reshape(-1,1)
    predicted = model.predict(x_train)
    f1score = metrics.f1_score(predicted,y_train)
 
    with open(os.path.join(output_model_path,'latestscore.txt'), 'w') as file:
        file.write(str(f1score))

    return f1score

if __name__ == '__main__':
    score_model('trainedmodel.pkl')


