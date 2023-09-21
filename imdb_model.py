import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

df_imdb = pd.read_csv('imdb_predictor_jupyter/movies.csv')

df_imdb = df_imdb.drop(df_imdb[df_imdb['Released_Year'] == 'PG'].index)
df_imdb['Released_Year'] = df_imdb['Released_Year'].astype(float)

df_imdb['Gross'] = df_imdb['Gross'].str.replace(',', "")
df_imdb_clean = df_imdb.dropna()

# RandomForestRegressor
X = pd.DataFrame()

X["Released_Year"] = df_imdb_clean['Released_Year']
# X["Runtime"] = df_imdb['Runtime']
X['Meta Score'] = df_imdb_clean['Meta_score']
X['Votes'] = df_imdb_clean['No_of_Votes']
X['Gross'] =  df_imdb_clean['Gross']


y = pd.DataFrame()
y['imdb'] = df_imdb_clean['IMDB_Rating']

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestRegressor(random_state=0)

model.fit(X_train, y_train)

pickle.dump(model, open('model_imdb.pkl','wb'))