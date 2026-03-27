
# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# To be used for data scaling and one hot encoding
from sklearn.preprocessing import StandardScaler

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load data set from Hugging face repo
DATASET_PATH = "hf://datasets/subhradasgupta/predictive_maintenance/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Separate features and target
X = df.drop(["Engine Condition"], axis=1)
y = df["Engine Condition"]

# Splitting data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

print("Created training and test dataset.")

# Perform standardization
sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Data preprossing done.")
print(X_train_df.shape, X_test_df.shape, y_train.shape, y_test.shape)

# Prepare input csv
input_df = X_test_df.tail(1)
input_df.to_csv("input.csv", index=False)

# Save splitted data locally
X_train_df.to_csv("Xtrain.csv", index=False)
X_test_df.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("Dataset saved locally.")

files = ["input.csv", "Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload the splitted data to Hugging Face
for file_path in files:
    print(f"Uploading file: {file_path} to HuggingFace")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="subhradasgupta/predictive_maintenance",
        repo_type="dataset",
    )
    print(f"Uploaded file to HuggingFace successfully")
