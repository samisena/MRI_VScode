import mlflow
import os

# Set credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = 'samisena'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cb6584b07822e49dbde524fb92b0376756fcfdfd'

# Try connecting
mlflow.set_tracking_uri("https://dagshub.com/samisena/MRI_VScode.mlflow")

# Test connection
experiments = mlflow.search_experiments()
print(experiments)