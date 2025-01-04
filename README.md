# Diabetes Multiclassification Documentation

![GitHub License](https://img.shields.io/github/license/Sambonic/diabetes-multiclassification)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

Diabetes multiclassification using various machine learning algorithms 
#### Last Updated: January 4th, 2025

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)

<a name="installation"></a>
## Installation

Make sure you have [python](https://www.python.org/downloads/) downloaded if you haven't already.
Follow these steps to set up the environment and run the application:

1. Clone the Repository:
   
```bash
git clone https://github.com/Sambonic/diabetes-multiclassification
```

```bash
cd diabetes-multiclassification
```

2. Create a Python Virtual Environment:
```bash
python -m venv env
```

3. Activate the Virtual Environment:
- On Windows:
  ```
  env\Scripts\activate
  ```

- On macOS and Linux:
  ```
  source env/bin/activate
  ```
4. Ensure Pip is Up-to-Date:
  ```
  python.exe -m pip install --upgrade pip
  ```
5. Install Dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Import Diabetes Multiclassification as shown below.


<a name="usage"></a>
## Usage

1.  **Ensure you have the necessary libraries installed**
2.  **Open the Jupyter Notebook (`diabetes_classification_ml.ipynb`)**.
3.  **Run all cells sequentially**. The notebook will output various plots and metrics showing the results of the analysis.  Note that the hyperparameter tuning sections are commented out; uncomment these sections to perform a full hyperparameter search for each model.  This will significantly increase the runtime.

The project uses a series of visualizations and analyses to build an effective model for predicting diabetes severity.  The results show the impact of different data preprocessing and feature selection methods on model performance.  The different model types show different characteristics and may be useful in different contexts.  The comparative analysis allows for an informed decision on which model is best suited for the given task.


<a name="features"></a>
## Features
* **Multiclass Diabetes Classification:** Predicts diabetes severity (no diabetes, prediabetes, diabetes) using machine learning.
* **Data Preprocessing:** Handles missing values using mean/mode imputation and outlier detection/adjustment for BMI.  Categorical features are handled using mode imputation and SMOTENC for oversampling.
* **Feature Engineering:** Discretizes BMI into meaningful categories.
* **Feature Selection:** Employs Chi-squared test and Random Forest feature importance to select relevant features.
* **Model Training:**  Trains and evaluates several classification models: Random Forest, Decision Tree, LightGBM, and Logistic Regression.  Learning curves are plotted for each model.
* **Model Evaluation:** Assesses model performance using accuracy, precision, recall, F1-score, confusion matrices, and ROC curves. Hyperparameter tuning is performed using RandomizedSearchCV to optimize model performance.
* **Class Imbalance Handling:** Addresses class imbalance through undersampling and SMOTENC oversampling techniques.
* **Comparative Analysis:** Compares the performance of different models with and without feature selection and hyperparameter optimization.  Results are visualized through bar charts.


