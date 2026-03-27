
# Libraries to help with reading and manipulating data

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

# To get different metric scores, and split data
from sklearn import metrics
from sklearn.metrics import (
    classification_report,
)

# To be used for data scaling and one hot encoding
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# To be used for tuning the model
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

import joblib
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

from huggingface_hub import hf_hub_download

# Reliable way to download items from huggingface.
def read_csv(file):
    print(f"Loading {file} from HuggingFace..")
    path = hf_hub_download(
        repo_id="subhradasgupta/predictive_maintenance",
        filename=file,
        repo_type="dataset",
    )

    df = pd.read_csv(path)
    print(f"{file} loaded.")
    return df


# Load the data sets
Xtrain = read_csv("Xtrain.csv")
Xtest = read_csv("Xtest.csv")
ytrain = read_csv("ytrain.csv")
ytest = read_csv("ytest.csv")

# This code will train the model using RandomizedSearchCV and also evaluate model performance
# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

model = XGBClassifier(random_state=1, eval_metric="logloss", n_jobs=-1)
param_grid = {
    "n_estimators": randint(100, 1000),
    "learning_rate": [0.01, 0.1],
    "gamma": uniform(0, 0.5),
    "max_depth": randint(3, 10),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
}

# Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=30,
    cv=5,
    random_state=1,
    n_jobs=-1,
    scoring=scorer,
)

print("Running RandomizedSearchCV ..")
# Fitting parameters in RandomizedSearchCV
randomized_cv.fit(Xtrain, ytrain)

results = randomized_cv.cv_results_

print("Best model selected.")

# Best model
best_model = randomized_cv.best_estimator_

print("Saving best model locally..")
# Save the model locally
model_path = "xgb_model_v1.joblib"
joblib.dump(best_model, model_path)
print("Model dump success.")

# Upload to Hugging Face
repo_id = "subhradasgupta/predictive_maintenance"
repo_type = "model"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

print("Uploading model to HuggingFace..")
# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")
# Step 2: Upload the model file to repo
api.upload_file(
    path_or_fileobj="xgb_model_v1.joblib",
    path_in_repo="xgb_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Model uploaded to HuggingFace.")
