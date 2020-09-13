import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle

df = pd.read_csv('./Telecom_customer churn cleaned.csv')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df.shape

df = df.iloc[:80, 40:]

#df = df.iloc[:80,40:100].drop(['Customer_ID'], axis = 1)
df = df.drop(['Customer_ID'], axis = 1)

df_dum = pd.get_dummies(df)
df_dum.head()

X = df_dum.drop('churn', axis = 1) 
y = df_dum.churn

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

state = 123
test_size = 0.30

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=state)

estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)

parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,
    verbose=True
)

grid_search.fit(X, y)

grid_search.best_estimator_

xgb_clf = grid_search.best_estimator_
xgb_clf.fit(X_train, y_train)
score = xgb_clf.score(X_val, y_val)
print(score)

pickl = {'model': grid_search.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

data_in = list(X_val[1,:])

model.predict(np.array(data_in).reshape(1,-1))[0]