import pickle
import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_folder_path = os.path.join(config['output_model_path'])
model_name = os.path.join(config['model_name'])
concat_file = os.path.join(config['concatfile'])

if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)


def init_logit_model() -> object:
    """This function will initialize the logistic regression model

    Returns:
        object: Logistic regression model
    """

    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None, max_iter=100,
                            multi_class='ovr', n_jobs=None, penalty='l2',
                            random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                            warm_start=False)
    return model

def train_model(model : object = None ,
                data_path: str = dataset_csv_path):
    """This function will train the model
    """

    # read the cleaned file from the output folder
    training_data = pd.read_csv(os.path.join(data_path,concat_file))

    # drop cooperation column
    training_data = training_data.drop(['corporation'], axis=1)

    # split the data into train and test
    x_train, _, y_train, _ = train_test_split(training_data.drop(['exited'], axis=1),
                                              training_data['exited'], test_size=0.2,
                                              random_state=0)

    if model is None:
        # use this logistic regression for training
        model = init_logit_model()

         # fit the logistic regression to your data
        model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)

    # write the trained model to your workspace in a file called
    pickle.dump(model, open(os.path.join(model_folder_path, model_name), 'wb'))

if __name__ == '__main__':
    train_model()
