# Customer Churn
Hero House AI Incubation Program 

- Build model to predict customer churn.
- Use Principal component analysis to understand which features are the most important.
- Optimized XGBoost using GridsearchCV to reach the best model.
- Deploy the model via FlaskAPI

### Data Cleaning

I made the following changes and created the following variables:
- Inputed `'Unknown'` for null values in categorical columns.
- Inputed the mode for null values in numerical columns.
- Treated outliers based on Z-score.
- Inputed the new mode for dropped outliers.

### Model Building

After cleaning the data, as the dataset has too many columns (100 before getting the dummy variables) I used Principal component analysis for selecting the best 3 columns (`'totmou', 'totcalls', 'totrev'`) for our model.
Next, I ran `XGBoost` algorithm with `GridSearchCV` in order to find the best hyperparameters.

### Results

The accuracy scores of the classifiers (Logistic Regression, k-nearest neighbors, and XGBoost) are a little greater than 50%. The project is still in progress.

### Resources

- Python Version: 3.8
- Packages: pandas, numpy, sklearn, matplotlib, seaborn, plotly, xgboost, pickle
- Video tutorial: https://www.youtube.com/watch?v=nUOh_lDMHOU
