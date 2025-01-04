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
To utilize this diabetes classification project:

1.  **Run the notebook:** Execute the Jupyter Notebook (`diabetes_classification_ml.ipynb`).  The notebook will perform the following actions automatically:

    *   Load the diabetes dataset.
    *   Perform exploratory data analysis (EDA), including data type checking, statistical analysis, visualization of missing values and class distributions, and correlation analysis.
    *   Handle missing values using mode imputation for categorical features and median imputation for numerical features. Evaluate different imputation methods.
    *   Handle outliers in the 'BMI' feature using IQR.
    *   Discretize the 'BMI' feature into meaningful categories.
    *   Balance the dataset using undersampling and oversampling techniques (SMOTENC).
    *   Perform feature selection using Chi-squared test and Random Forest feature importance.
    *   Train several classification models (Random Forest, Decision Tree, LightGBM, Logistic Regression) with and without feature selection and hyperparameter tuning.
    *   Evaluate model performance using various metrics (accuracy, precision, recall, F1-score) and visualize results using learning curves, confusion matrices, and ROC curves.
    *   Compare different model performance.

2.  **Interpret results:** The notebook will generate various visualizations and metrics that show the performance of different models under different conditions (with/without feature selection, with/without hyperparameter tuning).  Based on the results, one can determine which model performs best for diabetes classification.




<a name="features"></a>
## Features
- **Diabetes Multi-classification:** Predicts diabetes severity (no diabetes, pre-diabetes, diabetes) using machine learning.
- **Data Preprocessing:** Handles missing values using mean/mode imputation and outlier adjustments, and explores different imputation strategies (KNN, mean/median/mode).
- **Data Balancing:** Addresses class imbalance using undersampling of the majority class and oversampling of minority classes with SMOTENC (handling categorical features).
- **Feature Selection:** Employs Chi-squared test and Random Forest feature importance to select relevant features.
- **Model Training:** Trains and evaluates multiple classification models: Random Forest, Decision Tree, LightGBM, and Logistic Regression.
- **Model Evaluation:**  Uses various metrics (accuracy, precision, recall, F1-score) and visualizes results with confusion matrices and ROC curves.
- **Hyperparameter Tuning:** Optimizes model hyperparameters using `RandomizedSearchCV`.
- **Learning Curve Analysis:**  Plots learning curves to assess model bias and variance.
- **Comparative Analysis:**  Compares model performance with and without feature selection and hyperparameter tuning across multiple algorithms.


