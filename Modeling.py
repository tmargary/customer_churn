import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import pickle

df = pd.read_csv('./Telecom_customer churn cleaned.csv')#.sample(n = 5000) 

df.shape

#df = df.iloc[:80,40:100].drop(['Customer_ID'], axis = 1)
df = df.drop(['Customer_ID'], axis = 1)

df_dum = pd.get_dummies(df)
df_dum.head()




model = PCA(n_components=10).fit(df_dum)
X_pc = model.transform(df_dum)

# number of components
n_pcs= model.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = df_dum.columns
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
most_important_df = pd.DataFrame(dic.items())
most_important_df



df_pca = df_dum[['churn', 'totmou', 'totcalls', 'totrev']]

X = df_pca.drop('churn', axis = 1) 
y = df_pca.churn

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

state = 123
test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=state)


################### XGBClassifier ###################

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
score = xgb_clf.score(X_test, y_test)
print(score)

pickl = {'model': grid_search.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

data_in = list(X_test[1,:])

model.predict(np.array(data_in).reshape(1,-1))[0]


################### RandomForestClassifier ###################

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score

rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)

rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini')
rfc1.fit(X_train, y_train)

pred=rfc1.predict(X_test)

print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))


################### LogisticRegression ###################


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, np.ravel(y_train))
CMnorm = confusion_matrix(logreg.predict(X_test), y_test)
print(CMnorm)
print("The accuracy of Logistic regression: ", round(accuracy_score(y_test,logreg.predict(X_test)) * 100, 2))

################### naive_bayes ###################

from sklearn import naive_bayes

nbc = naive_bayes.GaussianNB()
nbc.fit(X_train,np.ravel(y_train))
CMnorm = confusion_matrix(nbc.predict(X_test), y_test)
print(CMnorm)
print("The accuracy of Naive Bayes: ", round(accuracy_score(y_test,nbc.predict(X_test)) * 100, 2))