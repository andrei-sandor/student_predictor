import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

df_covid = pd.read_csv("covid_predictor_jupyter/Cleaned-Data.csv")

df_covid = df_covid.drop(df_covid[df_covid["Age_20-24"] == 0].index)

#Logistic regression
X = pd.DataFrame()

X['Fever'] = df_covid["Fever"]
X['Tiredness'] = df_covid['Tiredness']
X["Dry-Cough"] = df_covid['Dry-Cough']
X["Difficulty-in-Breathing"] = df_covid["Difficulty-in-Breathing"]
X['Sore-Throat'] = df_covid["Sore-Throat"]
X['None_Symptom'] = df_covid['None_Sympton']
X['Pains'] = df_covid["Pains"]
X['Nasal-Congestion'] = df_covid['Nasal-Congestion']
X['Runny-Nose'] = df_covid['Runny-Nose']
X['Diarrhea'] = df_covid["Diarrhea"]
X["Age"] = df_covid["Age_20-24"]
X["Gender Female"] = df_covid["Gender_Female"]
X['Gender Male'] = df_covid['Gender_Male']
X['Gender_Transgender'] = df_covid["Gender_Transgender"]
X['idk Contact'] = df_covid['Contact_Dont-Know']
X['no contact'] = df_covid['Contact_No']
X['contact'] = df_covid["Contact_Yes"]

y = pd.DataFrame()
y['covid_mild'] = df_covid['Severity_None']

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
# Decision Tree Classifier

X = pd.DataFrame()

X['Fever'] = df_covid["Fever"]
X['Tiredness'] = df_covid['Tiredness']
X["Dry-Cough"] = df_covid['Dry-Cough']
X["Difficulty-in-Breathing"] = df_covid["Difficulty-in-Breathing"]
X['Sore-Throat'] = df_covid["Sore-Throat"]
X['None_Symptom'] = df_covid['None_Sympton']
X['Pains'] = df_covid["Pains"]
X['Nasal-Congestion'] = df_covid['Nasal-Congestion']
X['Runny-Nose'] = df_covid['Runny-Nose']
X['Diarrhea'] = df_covid["Diarrhea"]
X["Age"] = df_covid["Age_20-24"]
X["Gender Female"] = df_covid["Gender_Female"]
X['Gender Male'] = df_covid['Gender_Male']
X['Gender_Transgender'] = df_covid["Gender_Transgender"]
X['idk Contact'] = df_covid['Contact_Dont-Know']
X['no contact'] = df_covid['Contact_No']
X['contact'] = df_covid["Contact_Yes"]

y = pd.DataFrame()
y['covid_mild'] = df_covid['Severity_None']


model = GaussianNB()

model.fit(X, y)


pickle.dump(model, open('model_covid_mild.pkl','wb'))