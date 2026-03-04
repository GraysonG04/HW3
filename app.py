import streamlit as st
import boto3
import json
import pandas as pd

# 🔹 Replace with your actual endpoint name
ENDPOINT_NAME = "HW3-pipeline-endpoint-auto"
REGION = "us-east-1"

# Create SageMaker runtime client
runtime = boto3.client("sagemaker-runtime", region_name=REGION)

st.title("Bitcoin Buy / Hold / Sell Predictor")

st.write("Enter technical indicator values:")

# 🔹 Adjust these inputs to match your model features
roc_5 = st.number_input("ROC_5", value=0.0)

# If you kept more features, add them here:
# ema_5 = st.number_input("EMA_5", value=0.0)
# rsi_14 = st.number_input("RSI_14", value=50.0)

if st.button("Predict"):

    # Build input exactly as model expects
    input_data = pd.DataFrame([[roc_5]], columns=["ROC_5"])

    # Convert to JSON
    payload = json.dumps(input_data.values.tolist())

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload
    )

    result = json.loads(response["Body"].read().decode())

    # If classification model
    st.success(f"Prediction: {result}")
