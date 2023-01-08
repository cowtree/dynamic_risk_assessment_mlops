# collection of utility functions
import pickle

def load_ml_model(model_pathname: str):
    """This function will load the model

    Args:
        model_path (str): Path to the model

    Returns:
        model: Trained model
    """
    with open(model_pathname, 'rb') as file:
        model = pickle.load(file)
    return model
