import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
import torch
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer

def random_search(X,y, model, max_epochs, lr, optimizer, criterion, batch_size, callbacks, verbose, repeated_k_fold, 
                  param_grid, n_features, random_state = 42):
    """
    Perform randomized hyperparameter search for a neural network model.
    """
    n_features = X.shape[1]  

    net = NeuralNetRegressor(model, module__n_features = n_features, max_epochs=max_epochs, lr=lr, verbose=verbose,
                             optimizer=optimizer,criterion = criterion, callbacks = callbacks,batch_size = batch_size, device = "cuda")

    pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("float32", FunctionTransformer(
        lambda X: X.astype(np.float32)
    )),
    ("net", net)
    ])
    
    grid = RandomizedSearchCV(estimator = pipe,param_distributions=param_grid,scoring='neg_mean_absolute_error', 
                              n_jobs=-1,verbose=verbose, random_state = random_state, cv = repeated_k_fold)
    
    grid.fit(torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1))

    return grid

def get_best_hyper_parameters(grid):
    """
    Extract best hyperparameters from a fitted RandomizedSearchCV object.
    """
    cv_results = pd.DataFrame(grid.cv_results_)
    best_results = cv_results[cv_results["rank_test_score"] == 1]
    lr, wd, dp, batch = best_results["param_net__lr"].item(), best_results["param_net__optimizer__weight_decay"].item(), best_results["param_net__module__drop_out_rate"].item(), best_results["param_net__batch_size"].item()
    return lr, wd, dp, batch

def run_random_search(random_research_results_dfnn_loc, data, target, repeated_k_fold, param_grid, 
                      number_of_visual_features, prediction_head,random_state):
    # Load existing results or create empty dataframe
    if os.path.exists(random_research_results_dfnn_loc):
        random_search_results_dfnn = pd.read_csv(random_research_results_dfnn_loc)
    else:
        random_search_results_dfnn = pd.DataFrame(columns=["target", "lr", "wd", "dp", "bs"])
    
    # Check if valence already exists
    existing = random_search_results_dfnn[random_search_results_dfnn["target"] == target]
    
    if len(existing) > 0:
        # Retrieve stored values
        lr = existing.iloc[0]["lr"]
        wd = existing.iloc[0]["wd"]
        dp = existing.iloc[0]["dp"]
        batch = existing.iloc[0]["bs"]
    
        print("Using stored hyperparameters")
    
    else:
        print("Running random search")
        X = data[data.columns.difference(["image", "valence", "Unnamed: 0", "room_type","approach", "tense", "energetic"])]
        y = data[target]
        # Run random search
        grid = random_search(X, y, prediction_head,max_epochs=100,lr=0.001,optimizer=torch.optim.AdamW,criterion=torch.nn.L1Loss,
                             batch_size=216,callbacks=[EarlyStopping(monitor='valid_loss',patience=10,lower_is_better=True)], verbose=0,repeated_k_fold=repeated_k_fold,param_grid=param_grid,n_features=number_of_visual_features,random_state=random_state)
        # Extract all hyperparameters
        lr, wd, dp, batch = get_best_hyper_parameters(grid)
    
        # Append new result
        new_row = pd.DataFrame({"target": [target],"lr": [lr],"wd": [wd],"dp": [dp],"bs": [batch]})
    
        random_search_results_dfnn = pd.concat([random_search_results_dfnn, new_row],ignore_index=True)
    
        # Save to CSV
        random_search_results_dfnn.to_csv(random_research_results_dfnn_loc,index=False)

    return lr, wd, dp, int(batch)
