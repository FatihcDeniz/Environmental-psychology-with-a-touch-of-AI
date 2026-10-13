from utils.data_loaders import load_dataset
import shap
import numpy as np
import torch 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
def calculate_shap(model, train_dataloader, test_dataloader, feature_names, device):
    """
    Helper function to calculate SHAP values for DFNN and DCNN.
    """
    background_data = []
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        background_data.append(inputs)

    background_tensor = torch.cat(background_data)    

    test_inputs  = []
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        test_inputs.append(inputs)
        
    test_inputs = torch.cat(test_inputs) 
    
    model.eval()
    explainer = shap.GradientExplainer(model, background_tensor)
    shap_values = explainer(test_inputs)

    return shap_values, test_inputs, explainer

def plot_shap_summary_with_values(shap_values, X, feature_names=None, model_name="Model"):
    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Sort features by importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:10]
    sorted_features = X.columns[sorted_idx] if feature_names is None else [feature_names[i] for i in sorted_idx]
    sorted_means = mean_abs_shap[sorted_idx]

    # Format labels
    labels_with_values = [f"{feat} {val:.2f}" for feat, val in zip(sorted_features, sorted_means)]

    # Plot
    plt.title(model_name)
    shap.summary_plot(shap_values[:, sorted_idx], X.iloc[:, sorted_idx], feature_names=labels_with_values, show=False, plot_size=None)

def shap_dcnn_cross_validation(X, y, target, mean_data,repeated_k_fold):
    shap_all_folds = []
    feature_names = list(X.columns.difference(["image", "valence", "Unnamed: 0", "room_type", "approach", "tense", "energetic"]))
    for i, (train_idx, test_idx) in enumerate(repeated_k_fold.split(X, y)):
        print(f"Fold {i}")
        train_dataloader, test_dataloader, dataset = load_dataset(DataLoaderExperiment, mean_data, target, train_idx, test_idx, batch_size = 360)
        
        dcnn = DCNNHead(drop_out_rate=0.2, n_features = 768+number_of_visual_features)
        dcnn = dcnn.to(device)
        dcnn.load_state_dict(torch.load(f'/home/fdeniz2/models/single/c_dcnn_{target}_{i}.pth'))
        
        shap_values_dcnn, test_inputs_dcnn, dcnn_explainer = calculate_shap(dcnn, train_dataloader, test_dataloader, feature_names, device)
        shap_values_dcnn = np.abs(shap_values_dcnn.values).mean(axis=0)
        shap_all_folds.append(shap_values_dcnn)

    shap_all_folds = np.array(shap_all_folds) 

    return shap_all_folds, feature_names

def shap_dfnn_cross_validation(X, y, target,mean_data,repeated_k_fold):
    shap_all_folds = []
    feature_names = list(X.columns.difference(["image", "valence", "Unnamed: 0", "room_type", "approach", "tense", "energetic"]))
    for i, (train_idx, test_idx) in enumerate(repeated_k_fold.split(X, y)):
        print(f"Fold {i}")
        train_dataloader, test_dataloader, dataset = load_dataset(DataLoaderExperiment, mean_data, target, train_idx, test_idx, batch_size = 360)
        
        dfnn = DFNN(number_of_visual_features, 0.2).to(device)
        dfnn.load_state_dict(torch.load(f'/home/fdeniz2/models/dfnn/dfnn_{target}_{i}.pth'))
        
        shap_values_dfnn, test_inputs, dfnn_explainer = calculate_shap(dfnn, train_dataloader, test_dataloader, feature_names, device)
        X_test_dfnn = pd.DataFrame(test_inputs.detach().cpu().numpy(), columns=feature_names)
        shap_values_dfnn = np.abs(shap_values_dfnn.values).mean(axis=0)
        shap_all_folds.append(shap_values_dfnn)

    shap_all_folds = np.array(shap_all_folds) 

    return shap_all_folds, feature_names

def find_contribution(shap_values, feature_names, dcnn=False):
    color = ["Mean Hue", "Mean Saturation", "Mean Brightness", 
             "SD Saturation","SD Brightness", "SD Hue"]
    edge = ["Edge Density", "2nd order edge density"]
    complexity = ["Entropy", "2d_fractal_dimension"] 
    spatial_frequency = ["Fourier Slope", "Fourier Sigma"] 
    symmetry = ["Symmetry up-down", "Symmetry left-right"]
    objects = ['Plant','Light','Lamp','Cushion',"Flower", "Table", "Sofa", "Book", 
               'Painting, Picture','Cabinet','Chair','Bed','Window ']
    contour = ["Contour Curvature","Contour Length","Contour Orientation"]
    spatial_layout = ["Semi-enclosed area","Enclosed area",
                      "Far-away horizon", "Open area"] 
    functions = ["Playing", "Working","Exercise", "Cleaning", "Shopping", 
                 "Reading","Gaming", "Socializing"]
    
    categories = {"color": color,
        "edge": edge,
        "complexity": complexity,
        "spatial_frequency": spatial_frequency,
        "symmetry": symmetry,
        "objects": objects,
        "contour": contour,
        "spatial_layout": spatial_layout,
        "functions": functions}
    
    # Grouping into hierarchical levels
    low_level = ["color", "edge", "complexity", "spatial_frequency"]
    mid_level = ["symmetry", "objects", "contour", "spatial_layout"]
    high_level = ["functions"]

    def find_category(value):
        for category, values in categories.items():
            if value in values:
                return category
        return None  # not found

    #  Base contributions
    if dcnn:
        contributions = {k: 0 for k in categories.keys()}
        contributions["DCNN Features"] = 0
        mean_abs_shap = np.abs(shap_values).mean(axis=0).squeeze()
        for i in range(0, 41):
            cat = find_category(feature_names[i])
            if cat is not None:
                contributions[cat] += mean_abs_shap[i]
        contributions["DCNN Features"] = np.sum(mean_abs_shap[41:])

        for i in contributions.keys():
            if i != "DCNN Features":
                contributions[i] = contributions[i] / len(categories[i])
            else:
                contributions["DCNN Features"] = contributions["DCNN Features"] / 128
    
    else:
        contributions = {k: 0 for k in categories.keys()}
        mean_abs_shap = np.abs(shap_values).mean(axis=0).squeeze()
        for i in range(0, 41):
            cat = find_category(feature_names[i])
            if cat is not None:
                contributions[cat] += mean_abs_shap[i]

        for i in contributions.keys():
            contributions[i] = contributions[i] / len(categories[i])

    
    # Hierarchical contributions
    hierarchical = {
        "low_level": sum(contributions[c] for c in low_level),
        "mid_level": sum(contributions[c] for c in mid_level),
        "high_level": sum(contributions[c] for c in high_level),
    }

    return contributions, hierarchical

def plot_contributions(ax, contributions, contributions_group, extra_label=None):
    low_level = ["color", "edge", "complexity", "spatial_frequency"]
    mid_level = ["symmetry", "objects", "contour", "spatial_layout"]
    high_level = ["functions"]
    
    text_size = 7
    df = pd.DataFrame(list(contributions.items()), columns=["Category", "Contribution"])
    colors = []
    for cat in df["Category"]:
        if cat.lower() in low_level:
            colors.append(sns.color_palette("Greys", 9)[3])
        elif cat.lower() in mid_level:
            colors.append(sns.color_palette("Greys", 9)[5])
        elif cat.lower() in high_level:
            colors.append(sns.color_palette("Greys", 9)[7])
        elif extra_label and cat == extra_label:
            colors.append("black")
        else:
            colors.append(sns.color_palette("Greys", 9)[1])
    sns.barplot(data=df, x="Category", y="Contribution", palette=colors, ax=ax)
    ax.set_xticklabels(df["Category"], rotation=45, ha="right")
    max_contrib = df["Contribution"].max()
    ax.set_ylim(0, max_contrib * 1.18)
    for p, h in zip(ax.patches, df["Contribution"]):
        ax.text(p.get_x() + p.get_width()/2, h + max_contrib*0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9, color="black")

    group_counts = [
        len(low_level),
        len(mid_level),
        len(high_level)
    ]
    if extra_label:
        group_counts.append(1)

    cum_counts = [0] + [sum(group_counts[:i+1]) for i in range(len(group_counts))]
    positions = [(cum_counts[i] + cum_counts[i+1]-1)/2 for i in range(len(group_counts))]
    
    # ax.text(positions[0], max_contrib*1.1, f"Low: {round(contributions_group['low_level'].item(),2)}", ha="center", va="bottom", fontsize=text_size, color="black", fontweight='bold')
    # ax.text(positions[1], max_contrib*1.1, f"Mid: {round(contributions_group['mid_level'].item(),2)}", ha="center", va="bottom", fontsize=text_size, color="black", fontweight='bold')
    # ax.text(positions[2], max_contrib*1.1, f"High: {round(contributions_group['high_level'].item(),2)}", ha="center", va="bottom", fontsize=text_size, color="black", fontweight='bold')
    # if extra_label:
    #     ax.text(positions[3], max_contrib*1.1, f"DCNN: {round(contributions[extra_label].item(),2)}", ha="center", va="bottom", fontsize=text_size, color="black", fontweight='bold')

    for i in range(1, len(cum_counts)):
        ax.axvline(cum_counts[i]-0.5, color='black', linestyle='dotted', linewidth=1)
        
    low_end = len(low_level) - 0.5
    mid_end = low_end + len(mid_level)
    high_end = mid_end + len(high_level)
    ax.axvline(low_end, color='black', linestyle='dotted', linewidth=1)
    ax.axvline(mid_end, color='black', linestyle='dotted', linewidth=1)
    ax.axvline(high_end, color='black', linestyle='dotted', linewidth=1)
    if extra_label:
        extra_idx = df.index[df['Category'] == extra_label][0]
        ax.axvline(extra_idx + 0.5, color='black', linestyle='dotted', linewidth=1)


def plot_shap_summary_with_values(shap_values, X, feature_names=None, model_name="Model"):
    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Sort features by importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:10]
    sorted_features = X.columns[sorted_idx] if feature_names is None else [feature_names[i] for i in sorted_idx]
    sorted_means = mean_abs_shap[sorted_idx]

    # Format labels
    labels_with_values = [f"{feat} {val:.2f}" for feat, val in zip(sorted_features, sorted_means)]

    # Plot
    plt.title(model_name)
    shap.summary_plot(shap_values[:, sorted_idx], X.iloc[:, sorted_idx], feature_names=labels_with_values, show=False, plot_size=None)

def shap_summary_plot_all_models(shap_values_reg, X_test_reg, 
                                 shap_values_forest, X_test_forest,
                                 shap_values_dfnn, X_test_dfnn,
                                 shap_values_dcnn, X_test_dcnn,
                                 feature_names, feature_names_dcnn,
                                save, save_loc):
    all_shap_vals = np.concatenate([
        shap_values_reg.flatten(),
        shap_values_forest.flatten(),
        shap_values_dfnn.flatten(),
        shap_values_dcnn.flatten()
    ])
    max_abs_val = np.max(np.abs(all_shap_vals))
    ylim = [-max_abs_val, max_abs_val]
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(4, 1, 1)
    plot_shap_summary_with_values(shap_values_reg, X_test_reg, model_name="Linear Regression", feature_names = X_test_reg.columns) #X_test_reg
    plt.xlim(ylim)
    
    plt.subplot(4, 1, 2)
    plot_shap_summary_with_values(shap_values_forest, X_test_forest, model_name="Random Forest", feature_names = X_test_forest.columns)
    plt.xlim(ylim)
    
    plt.subplot(4, 1, 3)
    plot_shap_summary_with_values(shap_values_dfnn, X_test_dfnn,feature_names=X_test_dfnn.columns,model_name="DFNN")
    plt.xlim(ylim)
    plt.subplot(4, 1, 4)
    plot_shap_summary_with_values(shap_values_dcnn, X_test_dcnn,feature_names=X_test_dcnn.columns,model_name="C-DCNN")
    
    plt.xlim(ylim)
    plt.tight_layout()

    if save == True:
        plt.savefig(save_loc, format = "svg")
    plt.show()

def get_top_shap_names(X_test, shap_values,n):

    return list(X_test.columns[np.argsort(np.abs(shap_values).mean(0))[::-1]][:n])

def dependence_with_interaction(X_test, shap_values, save, save_loc,
                                n_features=10, per_fig=5, n_interactions=2,
                                fig_width=12, row_height=3):
    top_shap = list(get_top_shap_names(X_test, shap_values, n_features))
    base, ext = os.path.splitext(save_loc)
    n_figs = math.ceil(len(top_shap) / per_fig)

    for f in range(n_figs):
        chunk = top_shap[f * per_fig:(f + 1) * per_fig]
        fig, axes = plt.subplots(len(chunk), n_interactions,
                                 figsize=(fig_width, row_height * len(chunk)),
                                 squeeze=False)

        for r, feat in enumerate(chunk):
            feat_idx = X_test.columns.get_loc(feat)
            inter = shap.utils.approximate_interactions(feat_idx, shap_values, X_test)[:n_interactions]
            for c, inter_idx in enumerate(inter):
                ax = axes[r, c]
                shap.dependence_plot(feat_idx, shap_values, X_test,
                                     interaction_index=X_test.columns[inter_idx],
                                     ax=ax, show=False)
                ax.set_title(feat)

        fig.tight_layout()
        if save:
            fig.savefig(f"{base}_part{f + 1}{ext}", format="svg", bbox_inches="tight")
        plt.show()
        plt.close(fig)


