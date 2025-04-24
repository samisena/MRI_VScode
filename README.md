# Brain Tumour Detection with XAI

## Overview

This project implements an end-to-end machine learning solution for brain tumor detection using MRI scans. It focuses on both performance and interpretability by leveraging Explainable AI (XAI) techniques. By combining state-of-the-art CNN architectures with various explainability methods, this system not only achieves high accuracy in tumor detection but also provides insights into the decision-making process, making it valuable for clinical applications where transparency is crucial.

In this end-to-end ML project we will:

1. Prepare data for model training
2. Train 4 different CNN models using an MLOps methodology
3. Compare the models in terms of accuracy, computational complexity as well as interpretability
4. Deploy the models via REST API and a friendly user interface webpage

## Key Features

This project includes:

- Data processing and Model training with PyTorch
- Version Control with git
- Model version control with mlflow and DagsHub
- XAI techniques such as GradCAM, GuidedBP and Guided_GradCAM
- Model deployment using FastAPI
- Model monitoring over time with mlflow

## Results

The comprehensive analysis of results is available in the associated master's thesis. The project evaluated four different CNN architectures (ResNet50, ResNet101, EfficientNetB0, and EfficientNetB1) on various metrics including:

- Classification accuracy
- Precision, recall, and F1-score per class
- Computational efficiency (inference time, memory usage)
- Interpretability scores based on XAI visualization quality
- Model size and deployment considerations

## Data

The dataset was obtained from Kaggle: Brain Tumor MRI Dataset
This dataset contains MRI scans categorized into four classes:

1. Normal (no tumor)
2. Glioma
3. Meningioma
4. Pituitary tumor

The images have been preprocessed and standardized for model training.

## Models

The CNN model architectures implemented in this project are:

1. ResNet50: A deeper residual network with 50 layers
2. ResNet101: An extended version of ResNet with 101 layers
3. EfficientNetB0: A compact and efficient architecture
4. EfficientNetB1: A slightly larger variant of EfficientNetB0

## Repository Structure

This ML project follows a hybrid combination of both python files and jupyter notebooks. Functions are defined in Python files under the src/mri folder, but then used in notebooks under the Notebooks/ folder.
MRI_VSCODE/
├── Data/ # Contains the MRI dataset
├── Documentation/ # Project documentation
│ └── package_list.txt # List of required packages
├── monitoring/ # Model monitoring logs
├── Notebooks/ # Jupyter notebooks
│ ├── mlruns/ # Notebook-specific MLflow data
│ ├── Explainable.ipynb # XAI visualizations and analysis
│ ├── Exploratory.ipynb # Data exploration
│ └── Training.ipynb # Model training process
├── src/ # Source code
│ └── mri/ # Main package
│ ├── Deployment.py # FastAPI deployment script
│ ├── ExplainableAI.py # XAI implementation
│ ├── Model.py # Model architecture definitions
│ ├── Preprocessing.py # Data preprocessing functions
│ ├── Training.py # Training functions
│ └── utils.py # Utility functions
│ └── mri.egg-info/ # Package information
├── Templates/ # HTML templates
│ ├── monitoring.html # Dashboard for model monitoring
│ └── ui.html # User interface for model usage
├── environment.yml # Conda environment dependencies
├── model_monitoring.log # Monitoring logs
├── setup.py # Package setup script
└── README.md # This file

## Installation

This project requires Python 3.10 to run. It is recommended to use a conda virtual environment:
git clone https://github.com/yourusername/brain-tumour-detection.git
cd MRI_VSCODE
conda env create -f environment.yml
conda activate mri-env

## Usage

Running the Notebooks
The Jupyter notebooks provide a step-by-step walkthrough of the project:

- Exploratory.ipynb: Data analysis and visualization
- Training.ipynb: Training and evaluating models
- Explainable.ipynb: Applying XAI techniques to interpret model decisions

## Model Training

The training process can be monitored with MLflow:
mlflow ui

## Deployment

To deploy the model with FastAPI:
uvicorn mri.Deployment:app --reload

Once running, you can:
Use the web UI at http://localhost:8000/ to upload MRI images and get predictions
Monitor model performance at http://localhost:8000/monitoring
