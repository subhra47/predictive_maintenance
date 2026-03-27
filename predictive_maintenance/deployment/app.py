import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

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
input_df = read_csv("input.csv")
first_row = input_df.iloc[0]
input_dict = first_row.to_dict()

# Download and load the model
model_path = hf_hub_download(
    repo_id="subhradasgupta/predictive_maintenance", filename="xgb_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Engine Failure Prediction
st.title("Predictive Maintenance - Engine Failure Prediction app")
st.write(
    """Predict potential engine failure.
Please enter engine properties below to get a prediction.
"""
)

# User inputs
engine_rpm = st.number_input(
    "Engine rpm",
    value=input_dict.get("Engine rpm"),
)
lub_oil_pressure = st.number_input(
    "Lub oil pressure",
    value=input_dict.get("Lub oil pressure"),
)
fuel_pressure = st.number_input(
    "Fuel pressure",
    value=input_dict.get("Fuel pressure"),
)
coolant_pressure = st.number_input(
    "Coolant pressure",
    value=input_dict.get("Coolant pressure"),
)
lub_oil_temp = st.number_input(
    "lub oil temp",
    value=input_dict.get("lub oil temp"),
)
coolant_temp = st.number_input(
    "Coolant temp",
    value=input_dict.get("Engine rpm"),
)


# Assemble input into DataFrame
input_data = pd.DataFrame(
    [
        {
            "Engine rpm": engine_rpm,
            "Lub oil pressure": lub_oil_pressure,
            "Fuel pressure": fuel_pressure,
            "Coolant pressure": coolant_pressure,
            "lub oil temp": lub_oil_temp,
            "Coolant temp": coolant_temp,
        }
    ]
)


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Engine is Faulty" if prediction > 0.5 else "Engine is OK"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
