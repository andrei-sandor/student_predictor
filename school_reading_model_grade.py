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

df_read = df_full.drop(columns=["math score", "writing score"])

codes, uniques = pd.factorize(df_read["gender"])
df_read["gender factorized"] = np.array(codes)

codes_race, uniques_race = pd.factorize(df_read["race/ethnicity"])
df_read["race/ethnicity fact"] = np.array(codes_race)

codes_parental, uniques_parental = pd.factorize(df_read["parental level of education"])
df_read["parental level of education fact"] = np.array(codes_parental)

codes_lunch, unique_lunch = pd.factorize(df_read['lunch'])
df_read['lunch fact'] = np.array(codes_lunch)

codes_prep, uniques_prep = pd.factorize(df_read['test preparation course'])
df_read['prep fact'] = np.array(codes_prep)

# Random Forest Regressor
X = pd.DataFrame()

X["gender"] = df_read['gender factorized']
# X["race/ethnicity"] = df_math['race/ethnicity fact']
# X['parental'] = df_math["parental level of education fact"]
X['lunch'] = df_read["lunch fact"]
X['preparation'] = df_read["prep fact"]

y = pd.DataFrame()
y['math'] = df_read['reading score']

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestRegressor(random_state=0)

model.fit(X_train, y_train)

pickle.dump(model, open('model_school_reading_grade.pkl','wb'))