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
## Diabetes Multi-Classification Project Walkthrough

This project performs multi-class classification on a diabetes dataset to predict the severity of diabetes (no diabetes, pre-diabetes, diabetes).  The steps below outline the process, including data preprocessing, feature selection, model training, and evaluation.

**1. Data Loading and Exploration:**

The notebook begins by loading the diabetes dataset from a CSV file (`../datasets/diabetes_012_health_indicators_BRFSS2015.csv`). It then explores the data using descriptive statistics (`df.describe()`, `df.dtypes`, `df.nunique()`) and visualizes the distribution of features and their relationships with the target variable ('Diabetes_012') using histograms, box plots, count plots, and a correlation heatmap.  The code prints out the initial shapes, data types, and number of unique values for each feature.  It also identifies missing values and duplicates.

**2. Data Preprocessing:**

* **Handling Missing Values:** The notebook explores two imputation strategies:
    * **Mean/Mode Imputation:** Missing values in numerical features are filled with the mean, and those in categorical features are filled with the mode.  This is performed separately for the subset of classes {1, 2} and {0}.
    * **Dropping Missing Values:** Rows with missing values are dropped.

   The method achieving the highest accuracy using RandomForestClassifier on a subset of the data is selected.

* **Handling Duplicates:** Duplicate rows are removed from the dataset.

* **Outlier Handling:** Outliers in the 'BMI' feature are detected using the IQR method and are capped at the upper and lower bounds.

* **Discretization:** 'BMI' is discretized into meaningful categories (underweight, healthy weight, overweight, obesity classes).

* **Class Imbalance Handling:**  The dataset suffers from class imbalance. This is addressed with a combination of undersampling the majority class (class 0) and oversampling the minority classes (classes 1 and 2) using SMOTENC (to handle both categorical and numerical features).

**3. Feature Selection:**

Two feature selection methods are employed:

* **Chi-Squared Test:**  This statistical test identifies features most associated with the target variable. The top `k` features are selected based on their Chi-squared scores.

* **Random Forest Feature Importance:**  Feature importances are extracted from a trained Random Forest model. The top `k` features are selected based on these importances.

Finally, the intersection of features selected by both methods is used as the final set of features.

**4. Model Training and Evaluation:**

Several classification models are trained and evaluated: Random Forest, Decision Tree, LightGBM, and Logistic Regression.  For each model, the notebook performs the following:

* **Learning Curve Plot:** This visualizes the model's performance as a function of training data size to help determine if the model is overfitting or underfitting.

* **Model Training:** The model is trained on the training set.

* **Hyperparameter Tuning:** (Commented-out, likely due to time constraints) RandomizedSearchCV would have been used to find optimal hyperparameters for each model.  The code shows pre-defined parameters from the notebook author.

* **Model Evaluation:** The model is evaluated on the test set using accuracy, precision, recall, and F1-score. A confusion matrix and classification report are generated.

* **ROC Curve:**  ROC curves are plotted for each model, comparing its performance across different classes.

The results for all models, with and without feature selection and hyperparameter optimization are stored in a dictionary for further analysis.


**5. Results Comparison and Visualization:**

Finally, the notebook presents a comparison of the performance of different models using bar charts for each metric (accuracy, precision, recall, F1-score). The comparison is made for models with and without feature selection and hyperparameter tuning.  Separate comparisons are also shown for each model type (Random Forest, Decision Tree, LightGBM, Logistic Regression).


**To run the project:**

1.  **Ensure you have the necessary libraries installed:**  (Already done, per your assumption)
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


