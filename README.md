# Understanding affective responses to indoor environments using machine learning

Please read this before using the the Code. This folder contains the code and materials for "Environmental psychology with a touch of AI: understanding affective responses to indoor environments using machine learning". You can find the article [here](), and can contact the corresponding author by sending an email to `f.c.deniz@tue.nl`.

## Repository Structure

Before running the code please download IDEAS dataset from here: `https://osf.io/g9ze5`. You can unzip dataset in the samefolder. 

This repository seperated into multiple folders for processing the raw dataset and running the statistical analysis, explained below.

`Python Code`: this folder contains all scripts need to run LR, RF, DFNN, DCNN and C-DCNN models created in the study. You can run LR and RF model `LR_and_RF.ipynb`, DFNN model by running `DFNN.ipynb`, DCNN and C-DCNN model by running `DCNN.ipynb`. All SHAP analysis in `LR_and_RF.ipynb`. 

`Python Code\Utils`: this folder contains utility functions to run the code. `neural_network_utils.py` includes functions used to train DFNN, DCNN and C-DCNN models, `data_loaders.py` include functions to load dataset used in the study, `random_search_utils.py` include functions to conduct random search, `shap_utils.py` includes functions to calculate and visualize SHAP values and `utils.py` include various utility functions.

`Statiscal Analysis Code`: All statistical analyses conducted for the paper are documented in `Statistical Analysis Code\statistical_analysis.Rmd`, and all utility functions used in these analyses are provided in `utils.R`. Running or knitting `statistical_analysis.Rmd` reproduces all statistical analyses reported in the paper and generates all tables and figures included in the paper.

`Models` folder includes all DFNN, DCNN, and C-DCNN models trained in the study.

`Results`: this folder contains results from all machine learning models. 

## References

