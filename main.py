
# Import pandas to convert excel data to dataframe
import pandas as pd

# Import the train-test split, cross validation functions, and the Decision Tree Regressor model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor

# Import regression evaluation metrics to assess algorithm performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read csv data and convert to dataframe
df = pd.read_csv("Employee Attrition.csv")

# Clean and preprocess data by removing unnecessary columns/values, removing duplicates, handling missing values,  and encoding categorical values
df = df.drop(columns='Emp ID')
df = df.drop_duplicates()

df["salary"] = df["salary"].map({
    "low": 0,
    "medium": 1,
    "high": 2
})

df = pd.get_dummies(df, columns=["dept"], dtype=int)
df = df.dropna()

# Export cleaned data to csv
df.to_csv("Employee Attrition Preprocessed.csv", index=False)

# Separate the feature variables and the target variable
X = df.drop(columns=['satisfaction_level'])
y = df['satisfaction_level']

# Split the data between training (80%) and testing (20%) sets with random state set for consistency between runs
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size= 0.2)

# Create and train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=17)
model.fit(X_train, y_train)

# Use the trained model to predict employee satisfaction levels for the test data
pred = model.predict(X_test)

# Assess the baseline algorithm performance with evaluation metrics
print(f"\n(Baseline decision tree)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, pred)}\n"
      f"Mean Absolute Error: {mean_absolute_error(y_test, pred)}\n"
      f"R² score: {r2_score(y_test, pred)}\n")

# Apply cross-validation evaluate how the model performs across different training subsets
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

print("\nCross-Validation\n"
f"R² Scores: {cv_scores}\n"
f"Average R²: {cv_scores.mean()}\n")

# Apply Grid Search and define hyperparameter values for it to test 
param_grid = [{
    'max_depth': [3, 5, 7, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10]
}]

dt_grid = GridSearchCV(model,
                        cv=5,
                        param_grid=param_grid,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
)

dt_grid.fit(X_train, y_train)

# Use the optimized Decision Tree model to predict employee satisfaction levels
dt_grid_pred = dt_grid.best_estimator_.predict(X_test)

# Assess the optimized algorithm performance with the same evaluation metrics
print(f"\n(Optimized decision tree)\n" f"Mean Squared Error: {mean_squared_error(y_test, dt_grid_pred)}\n" 
      f"Mean Absolute Error: {mean_absolute_error(y_test, dt_grid_pred)}\n" 
      f"R² score: {r2_score(y_test, dt_grid_pred)}\n")


