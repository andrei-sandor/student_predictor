import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
import pickle


df_math = pd.read_excel("school_predictor_jupyter/student-math.xlsx")

codes_gender, uniques_gender = pd.factorize(df_math["Gender"])
df_math["gender factorized"] = np.array(codes_gender)

codes_fam, uniques_fam = pd.factorize(df_math["famsize"])
df_math["fam size factorized"] = np.array(codes_fam)

codes_activities, uniques_activities = pd.factorize(df_math["activities"])
df_math["activities factorized"] = np.array(codes_activities)



X = pd.DataFrame()

X["gender"] = df_math['gender factorized']
X["age"] = df_math['age']
X["fam_size_Fact"] = df_math['fam size factorized']
X['travel'] = df_math['traveltime']
X['study'] = df_math["studytime"]
X['activities'] = df_math['activities factorized']
X['free'] = df_math['freetime']
X['out'] = df_math['goout']
X['absences'] = df_math['absences']                     

y = pd.DataFrame()
y['mid1'] = df_math['G1']

model = RandomForestRegressor(random_state=0)

model.fit(X, y)


pickle.dump(model, open('model_math_mid1.pkl','wb'))
