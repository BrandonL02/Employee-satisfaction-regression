# Employee Satisfaction Prediction

This project uses a **Decision Tree Regressor** to predict employee satisfaction levels from employee workplace data. The project includes data preprocessing, baseline model evaluation, cross-validation, and hyperparameter tuning using Grid Search.

## Dataset

Dataset source: [Employees Satisfaction Analysis - Kaggle](https://www.kaggle.com/datasets/redpen12/employees-satisfaction-analysis)

## Project Overview

The application:

- Cleans and preprocesses the employee dataset
- Encodes categorical variables for machine learning
- Splits the data into training and testing sets
- Trains a baseline Decision Tree Regressor
- Evaluates the model using regression metrics
- Performs 5-fold cross-validation
- Uses Grid Search to optimize model hyperparameters
- Compares the baseline and optimized models
- Exports the cleaned dataset as `Employee Attrition Preprocessed.csv`

## Model

**Decision Tree Regressor**

The model predicts:

`Satisfaction Level`

Hyperparameters optimized with `GridSearchCV`:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

## Evaluation

Model performance is evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- 5-Fold Cross-Validation

## Data Preprocessing

Before model training, the application:

- Removes the `Emp ID` column
- Removes duplicate records
- Encodes salary levels as numerical values
- One-hot encodes the `dept` column
- Removes rows containing missing values

## Requirements

### Software

- Windows 11
- Python 3.13
- Visual Studio Code or equivalent Python IDE

### Libraries

- pandas 2.3.3
- scikit-learn 1.8.0

## Installation

Clone the repository:

```bash
git clone https://github.com/BrandonL02/Employee-satisfaction-regression.git
```

## Author

**Brandon Latimer**

B.S. Computer Science
