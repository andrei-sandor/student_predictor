import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection
import pickle

df_full = pd.read_csv("school_predictor_jupyter/StudentsPerformance.csv")

df_write = df_full.drop(columns=["math score", "reading score"])

codes, uniques = pd.factorize(df_write["gender"])
df_write["gender factorized"] = np.array(codes)

codes_race, uniques_race = pd.factorize(df_write["race/ethnicity"])
df_write["race/ethnicity fact"] = np.array(codes_race)
codes_parental, uniques_parental = pd.factorize(df_write["parental level of education"])

df_write["parental level of education fact"] = np.array(codes_parental)

codes_lunch, unique_lunch = pd.factorize(df_write['lunch'])
df_write['lunch fact'] = np.array(codes_lunch)

codes_prep, uniques_prep = pd.factorize(df_write['test preparation course'])
df_write['prep fact'] = np.array(codes_prep)

df_write.loc[df_write["writing score"] < 60, "pass/fail"] = 0

df_write.loc[df_write["writing score"] >= 60, "pass/fail"] = 1

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# Decision Tree Classifier
X = pd.DataFrame()

X["gender"] = df_write['gender factorized']
# X["race/ethnicity"] = df_math['race/ethnicity fact']
# X['parental'] = df_math["parental level of education fact"]
X['lunch'] = df_write["lunch fact"]
X['preparation'] = df_write["prep fact"]

y = pd.DataFrame()
y['math'] = df_write['pass/fail']

# smt = SMOTE()
# X_train_sm, y_train_sm = smt.fit_resample(X, y)
# X_train, x_test, y_train, y_test = train_test_split(X_train_sm, y_train_sm, test_size=0.2, random_state=0)

oss = OneSidedSelection()
X_train_oss, y_train_oss = oss.fit_resample(X, y)
X_train, x_test, y_train, y_test = train_test_split(X_train_oss, y_train_oss, test_size=0.2, random_state=0)

# X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

pickle.dump(model, open('model_school_writing_pass_fail.pkl','wb'))