# Repo for article "NeuralSympCheck: A Symptom Checking and Disease Diagnostic Neural Model with Logic Regularization"

## Overview

This archive contains code to reproduce the main results described in the article "NeuralSympCheck: A Symptom Checking and Disease Diagnostic Neural Model with Logic Regularization". 

To reproduce these results, follow the steps described below.

## 1. Clone this repository

## 2. Environment

Create an isolated environment with venv or conda using python==3.7.10. Install the dependencies listed in the src/requirements.txt

## 3. Download datasets

Download datasets from https://drive.google.com/drive/folders/19Jv_4wwC6LM485hDf8O5uhYb8QbAZOms?usp=sharing and put its to the folder data/05_model_input

## 4. Run pipelines

Activate the environment and run this command at the root of the repository: 
    kedro run --pipeline symptom_checker --params device:*device*,ds_name:*ds_name*,mode:test

Where:
- *device* - GPU idx or 'cpu'
- *ds_name*:
    - mz - for MuZhi dataset
    - dxy - for Dxy dataset
    - symcat_200 - for SymCat dataset with 200 deseases
    - symcat_300 - for SymCat dataset with 300 deseases
    - symcat_400 - for SymCat dataset with 400 deseases

The following results will be saved after the Pipeline is completed:
- model in the folder data/06_models
- metrics in the folder data/08_reporting
