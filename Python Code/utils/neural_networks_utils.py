import os 
import torch
import torch.nn as nn
import numpy as np
import copy
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error, r2_score
from scipy.stats import  pearsonr
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEma
from skorch.callbacks import EarlyStopping

from utils.data_loaders import load_dataset, load_csv
from utils.utils import to_dataframe_
from pathlib import Path
from torchvision import models
import pandas as pd
from torchvision.models.feature_extraction import create_feature_extractor


class prediction_head(nn.Module):
    """
    Deep Feedforward Neural Network with three hidden layers,
    LayerNorm, GELU activations, and dropout.
    """
    def __init__(self, n_features, drop_out_rate=0.25, network = None):
        super().__init__()
        if network:
            self.network = network
        else:
            self.network = nn.Sequential(
                nn.Linear(n_features, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(drop_out_rate),
    
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(drop_out_rate),
    
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
    
                nn.Linear(128, 1))

    def forward(self, x):
        return self.network(x)


def get_feature_columns(mean_data):
    """
    Returns the columns used as DFNN input features.
    """

    non_feature_cols = [
        "image",
        "Unnamed: 0",
        "room_type",
        "valence",
        "tense",
        "energetic",
        "approach"
    ]

    return mean_data.columns.difference(non_feature_cols)


def scale_fold_data(mean_data, train_idx, test_idx, feature_cols):
    """
    Scale features using only the training fold.
    The test fold is transformed using the training-fold scaler.
    """
    from sklearn.preprocessing import StandardScaler

    scaled_data = mean_data.copy()
    # Get train and test data from cross-validation indexes
    train_index = mean_data.index[train_idx]
    test_index = mean_data.index[test_idx]
    # Scaler
    scaler = StandardScaler()
    # Scale both train and test data
    scaled_data.loc[train_index, feature_cols] = scaler.fit_transform(mean_data.loc[train_index, feature_cols].astype(np.float32))
    scaled_data.loc[test_index, feature_cols] = scaler.transform(mean_data.loc[test_index, feature_cols].astype(np.float32))
    return scaled_data, scaler


def train_single_model(model,optimizer,criterion,scheduler,train_dataloader,test_dataloader, num_epochs, device, patience, min_delta=0.0):
    """
    Train one DFNN model with early stopping and cosine LR scheduling.
    """
    # Create lists to store test and train lossess
    train_losses = []
    test_losses = []
    # Create variables to store losses
    best_test_loss = np.inf
    best_state_dict = copy.deepcopy(model.state_dict())
    best_r2 = None
    best_mae = None
    best_epoch = 0

    epochs_without_improvement = 0
    # Loop through all epochs
    for epoch in range(num_epochs):
        # Set the model for training
        model.train()

        train_loss_sum = 0.0
        train_n = 0
        # Loop through the data
        for inputs, labels in train_dataloader:
            # Move data to gpu
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            optimizer.step()
            # Train loss for the epoch
            batch_size = inputs.size(0)
            train_loss_sum += loss.item() * batch_size
            train_n += batch_size
        # Store train loss
        train_loss = train_loss_sum / train_n
        train_losses.append(train_loss)

        # Put model on eval mode
        model.eval()
        
        test_loss_sum = 0.0
        test_n = 0
        predicted_batches = []
        actual_batches = []
        # Run model to calculate and store test loss
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1)

                outputs = model(inputs).view(-1)
                loss = criterion(outputs, labels)

                batch_size = inputs.size(0)
                test_loss_sum += loss.item() * batch_size
                test_n += batch_size

                predicted_batches.append(outputs.detach().cpu())
                actual_batches.append(labels.detach().cpu())
        # Calculate and store test loss
        test_loss = test_loss_sum / test_n
        test_losses.append(test_loss)
        # Calculate test MAE and R2
        predicted_values = torch.cat(predicted_batches).numpy()
        actual_values = torch.cat(actual_batches).numpy()
        mae = mean_absolute_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)

        # Check whether test loss improved and update the model and patience
        improved = test_loss < (best_test_loss - min_delta)
        if improved:
            best_test_loss = test_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_r2 = r2
            best_mae = mae
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            print("Best model updated")
        else:
            epochs_without_improvement += 1

        scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.5f} | "
            f"Test Loss: {test_loss:.5f} | "
            f"MAE: {mae:.5f} | "
            f"R2: {r2:.5f} | "
            f"Patience: {epochs_without_improvement}/{patience} | "
            f"LR: {current_lr:.8f}"
        )

        # Early stopping if patience is reached
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    model.load_state_dict(best_state_dict)

    return model, train_losses, test_losses, best_r2, best_mae, best_epoch


def train_model_cv(model_function,mean_data,repeated_k_fold,target, lr, wd, dp, batch, number_of_visual_features, n_epochs, 
                   patience, device, save=True, save_loc=None, multiple_folder = True):
    """
    Train DFNN models using cross-validation with fold-wise feature scaling.
    """
    # Check number of features
    feature_cols = get_feature_columns(mean_data)
    
    if len(feature_cols) != number_of_visual_features:
        raise ValueError(f"number_of_visual_features={number_of_visual_features}, "f"but detected {len(feature_cols)} feature columns.")
    # Check save location if it doesn't exist create one
    if save:
        if save_loc is None:
            raise ValueError("save_loc must be provided when save=True.")
        save_loc = Path(save_loc)
        save_loc.mkdir(parents=True, exist_ok=True)
    # Store all the results
    k_folds_results = {
        "model": [],
        "train_loss": [],
        "test_loss": [],
        "best_r2": [],
        "best_mae": [],
        "best_epoch": []
    }
    # Loop through all cross-validation splits
    for fold, (train_idx, test_idx) in enumerate(repeated_k_fold.split(mean_data)):
        # Load the model
        model = model_function(n_features=number_of_visual_features, drop_out_rate=dp).to(device)
        print(f"\nTraining fold {fold}")
        # Scale train and test data
        scaled_fold_data, scaler = scale_fold_data(mean_data=mean_data,train_idx=train_idx,test_idx=test_idx,feature_cols=feature_cols)
        # Load data in batches
        train_dataloader, test_dataloader, dataset = load_dataset(load_csv,multiple_folder,scaled_fold_data,target,
                                                                  train_idx,test_idx,batch_size=batch)
        # MAE loss
        criterion = nn.L1Loss()
        # AdamW optimizer
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=wd)
        # Cosine LR Scheduler
        scheduler = CosineLRScheduler(optimizer,t_initial=n_epochs, warmup_t=min(20, n_epochs),
                                      warmup_lr_init=lr / 10, lr_min=lr / 100, cycle_limit=1)

        # Train the model for single cross-validation split
        model, train_losses, test_losses, best_r2, best_mae, best_epoch = train_single_model(model=model,optimizer=optimizer,criterion=criterion,scheduler=scheduler, 
                                                                                             train_dataloader=train_dataloader, test_dataloader=test_dataloader, num_epochs=n_epochs, 
                                                                                             device=device, patience=patience)
        
        # Store the results from the cross-validation split
        model_name = f"{target}_{fold}"
        k_folds_results["model"].append(model_name)
        k_folds_results["train_loss"].append(np.mean(train_losses))
        k_folds_results["test_loss"].append(np.mean(test_losses))
        k_folds_results["best_r2"].append(best_r2)
        k_folds_results["best_mae"].append(best_mae)
        k_folds_results["best_epoch"].append(best_epoch)
        # Save the model
        if save:
            save_file = save_loc / f"{model_name}.pth"
            torch.save(model.state_dict(),save_file)
            print(f"Model saved f{save_file}")
    return k_folds_results

def performance_model_cv(model_function, mean_data, repeated_k_fold, target, device, n_features, drop_out_rate, 
                         model_dir="./models/DFNN", batch_size=256, scale=True, save=True, save_loc="./results/DFNN", 
                         multiple_folder = True):
    """
    Evaluate trained DFNN models using cross-validation metrics.

    Important:
    - Pass the original unscaled mean_data.
    - If scale=True, scaling is done inside each CV fold.
    - The scaler is fitted only on the training fold.
    """

    model_dir = Path(model_dir)

    feature_cols = get_feature_columns(mean_data)

    if len(feature_cols) != n_features:
        raise ValueError(f"n_features={n_features}, but detected {len(feature_cols)} feature columns.")

    model_performance_dfnn = {
        "RMSE": [],
        "MAE": [],
        "Rsquared": [],
        "PCC": []
    }
    # Loop through all cross-validation splits
    for fold, (train_idx, test_idx) in enumerate(repeated_k_fold.split(mean_data)):
        print(f"Evaluating fold {fold}")
        # Scale the test data
        if scale:
            fold_data, _ = scale_fold_data(mean_data=mean_data,train_idx=train_idx,test_idx=test_idx,
                                           feature_cols=feature_cols)
        else:
            fold_data = mean_data.copy()

        # Load test data
        _, test_dataloader, _ = load_dataset(load_csv,multiple_folder,fold_data,target, 
                                             train_idx,test_idx, batch_size=batch_size)


        # Load the model
        model = model_function(n_features=n_features, drop_out_rate=drop_out_rate).to(device)
        model_path = model_dir/ f"{target}_{fold}.pth" 
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        predicted_batches = []
        actual_batches = []
        
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1)

                outputs = model(inputs).view(-1)

                predicted_batches.append(outputs.detach().cpu())
                actual_batches.append(labels.detach().cpu())
        # Predicted and actual values
        predicted_values = torch.cat(predicted_batches).numpy()
        actual_values = torch.cat(actual_batches).numpy()
        # Calculate metrics
        rmse = root_mean_squared_error(actual_values, predicted_values)
        mae = mean_absolute_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)
        pcc = pearsonr(actual_values, predicted_values).statistic
        # Store metrics
        model_performance_dfnn["RMSE"].append(rmse)
        model_performance_dfnn["MAE"].append(mae)
        model_performance_dfnn["Rsquared"].append(r2)
        model_performance_dfnn["PCC"].append(pcc)
    
    # Convert scores to a pandas dataframe
    model_performance_dfnn = to_dataframe_(model_performance_dfnn, str(model_dir).split("/")[-1])
    # Save scores to csv
    if save:
        model_performance_dfnn.to_csv(save_loc)

    return model_performance_dfnn

def load_convx(device):
    # Load Model
    convx = models.convnext_tiny(pretrained = True)
    # Freeze model
    for param in convx.features.parameters():
        param.requires_grad = False

    convx = convx.to(device)
    return convx

def extract_embeddings(model, return_nodes, dataset, device):
    layer_1_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    # Extract Features
    all_outputs = []
    all_labels = []
    image_names = []
    for inputs, image_name in dataset:
        inputs = inputs.to(device)
        outputs = layer_1_extractor(inputs)
        all_outputs.append(outputs["layer"].flatten(start_dim = 1).detach().cpu())

        image_name = [i.split("/")[-1] for i in image_name]
        image_names.extend(image_name)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    
    all_outputs = pd.DataFrame(all_outputs.numpy())
    all_outputs["image"] = image_names

    return all_outputs