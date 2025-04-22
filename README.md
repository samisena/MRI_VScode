# Brain Tumour Detection with XAI

## Overview

## Brief description of the project, its purpose, and the problem it solves.

This end-to-end ML project consists of training 4 different CNN architectures on
a brain MRI images dataset to detect and classify tumours.
The models are then compared not only by accuracy levels achieved but also by the
quality of their interpretability with the modern Explainable AI techniques.

## Key Features

- This project includes MLOps practices such as: model version control with mlflow and data version control on DagsHub. As well as model deployment via REST API and model monitoring.
- Beyond metrics of accuracy and computational complexity - the CNN model's a interpretability is also evaluated using explainable algorithms: GradCAM, GuidedBP and Guided_GradCAM.

## Results

A brief summary of results with key metrics and possibly a visualization.

## Data

The datasets were obtained from Kaggle:

## Models

The CNN model architectures are: Resnet50, Resnet101, Efficientnetb0 and Efficientnetb1.

## Repository Structure

This ML project follows a hybrid combination of both python files and jupyter notebooks. Functions are defined in Python files under the src\mri folder, but then used in notebooks under the Notebooks\ folder.

## Installation

This project requires python 3.10 to run.

```bash
pip install -r requirements.txt
```
