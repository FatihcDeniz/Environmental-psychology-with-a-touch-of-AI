import torch
from torch import nn
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error, r2_score
import numpy as np
from scipy.stats import  pearsonr
from torchvision.models.feature_extraction import create_feature_extractor
import seaborn as sns

def fix_names_dataframe(mean_data):
    name_mapping = {
        'mean_h': 'Mean Hue',
        'mean_s': 'Mean Saturation',
        'mean_b': 'Mean Brightness',
        'std_s': 'SD Saturation',
        'std_h': 'SD Hue',
        'std_b': 'SD Brightness',
        'fourier_slope': 'Fourier Slope',
        'fourier_sigma': 'Fourier Sigma',
        'edge_density': 'Edge Density',
        'entropy': 'Entropy',
        'second_order_ed':'2nd order edge density',
        'first_order_ed':'1st order edge density',
        'table': 'Table',
        'swivel chair': 'Swivel Chair',
        'chair': 'Chair',
        'lamp': 'Lamp',
        'sink': 'Sink',
        'stove': 'Stove',
        'bed': 'Bed',
        'computer': 'Computer',
        'cushion': 'Cushion',
        'desk': 'Desk',
        'light': 'Light',
        'cabinet': 'Cabinet',
        'painting, picture': 'Painting',
        'sofa': 'Sofa',
        'window ': 'Window',
        'open area': 'Open area',
        'semi-enclosed area': 'Semi-enclosed area',
        'far-away horizon': 'Far-away horizon',
        'enclosed area': 'Enclosed area',
        'symmetry_ud': 'Symmetry up-down',
        'symmetry_lr': 'Symmetry left-right',
        'curvature7': 'Contour Curvature',
        'length7': 'Contour Length',
        'orientation5': 'Contour Orientation',
        'reading': 'Reading',
        'using tools': 'Using tools',
        'exercise': 'Exercise',
        'playing': 'Playing',
        'working': 'Working',
        'cleaning': 'Cleaning',
        'gaming': 'Gaming',
        'socializing': 'Socializing',
        'shopping': 'Shopping',
    }
    return mean_data.rename(columns  = name_mapping, inplace = True)

def check_create_directory(path: str) -> None:
    import os
    # Check a directory exist otherwise create one.
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


def fit_models_cross_validation(X, y, model, repeated_k_fold, explainer_class, check_additivity, calculate_shap):
    model_performance = {"RMSE":[], "MAE":[], "Rsquared": [], "PCC":[]}
    feature_names = X.columns
    shap_all_folds = []
    
    for i, (train_idx, test_idx) in enumerate(repeated_k_fold.split(X, y)):
        #print(f"Fold {i}")
        # Create the split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        # Fit the model
        model.fit(X_train, y_train)
        predicted_values = model.predict(X_test)
        # Calculate Performance
        model_performance["Rsquared"].append(r2_score(y_test, predicted_values))
        model_performance["MAE"].append(mean_absolute_error(y_test, predicted_values))
        model_performance["PCC"].append(pearsonr(y_test, predicted_values)[0])
        model_performance["RMSE"].append(np.sqrt(mean_squared_error(y_test, predicted_values)))
        # Calculate Shap

        if calculate_shap == True:
            explainer = explainer_class(model, X_train)
            if check_additivity == False:
                shap_values = explainer.shap_values(X_test, check_additivity = check_additivity)
            else:
                shap_values = explainer.shap_values(X_test)
            # mean absolute SHAP per feature for this fold
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
            shap_all_folds.append(mean_abs_shap)
        else:
            shap_all_folds = 0
        
    shap_all_folds = np.array(shap_all_folds) 
    
    return model_performance, shap_all_folds, feature_names 

def detach_tensor(tensor):
    """
    Helper function to detach tensor.
    """
    return tensor.detach().cpu().numpy()

def to_dataframe_(model_performance, model):
    """
    Convert model performance dictionary to a DataFrame and tag model name.

    Parameters
    ----------
    model_performance : dict
        Dictionary containing evaluation metrics.
    model : str
        Name/identifier of the model.

    Returns
    -------
    pd.DataFrame
        DataFrame with performance metrics and model column.
    """
    model_performance = pd.DataFrame(model_performance)
    model_performance["Model"] = model
    model_performance[["RMSE", "MAE", "Rsquared", "Model"]]
    return model_performance

