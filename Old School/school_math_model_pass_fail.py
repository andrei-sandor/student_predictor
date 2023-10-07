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

df_math = df_full.drop(columns=["reading score", "writing score"])

codes, uniques = pd.factorize(df_math["gender"])

df_math["gender factorized"] = np.array(codes)

codes_race, uniques_race = pd.factorize(df_math["race/ethnicity"])

df_math["race/ethnicity fact"] = np.array(codes_race)

codes_parental, uniques_parental = pd.factorize(df_math["parental level of education"])

df_math["parental level of education fact"] = np.array(codes_parental)

codes_lunch, unique_lunch = pd.factorize(df_math['lunch'])
df_math['lunch fact'] = np.array(codes_lunch)

codes_prep, uniques_prep = pd.factorize(df_math['test preparation course'])
df_math['prep fact'] = np.array(codes_prep)

df_math.loc[df_math["math score"] < 60, "pass/fail"] = 0
df_math.loc[df_math["math score"] >= 60, "pass/fail"] = 1

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# Decision Tree Classifier
X = pd.DataFrame()

X["gender"] = df_math['gender factorized']
# X["race/ethnicity"] = df_math['race/ethnicity fact']
# X['parental'] = df_math["parental level of education fact"]
X['lunch'] = df_math["lunch fact"]
X['preparation'] = df_math["prep fact"]

y = pd.DataFrame()
y['math'] = df_math['pass/fail']

# smt = SMOTE()
# X_train_sm, y_train_sm = smt.fit_resample(X, y)
# X_train, x_test, y_train, y_test = train_test_split(X_train_sm, y_train_sm, test_size=0.2, random_state=0)

oss = OneSidedSelection()
X_train_oss, y_train_oss = oss.fit_resample(X, y)
X_train, x_test, y_train, y_test = train_test_split(X_train_oss, y_train_oss, test_size=0.2, random_state=0)

# X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

pickle.dump(model, open('model_school_math_pass_fail.pkl','wb'))