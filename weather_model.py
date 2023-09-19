import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pickle

df_meteo = pd.read_csv("meteo/weather.csv")

df_meteo = df_meteo.drop(df_meteo[df_meteo['precipitation'] > 25].index)
std_precipitations = df_meteo['precipitation'].std()
mean_precipitations = df_meteo['precipitation'].mean()
df_meteo = df_meteo.drop(df_meteo[df_meteo['precipitation'] > (mean_precipitations + 3*std_precipitations)].index)

std_tmax = df_meteo['temp_max'].std()
mean_tmax = df_meteo['temp_max'].mean()
df_meteo = df_meteo.drop(df_meteo[df_meteo['temp_max'] > (mean_tmax + 3*std_tmax)].index)
df_meteo = df_meteo.drop(df_meteo[df_meteo['temp_max'] < (mean_tmax - 3*std_tmax)].index)

std_tmin = df_meteo['temp_min'].std()
mean_tmin = df_meteo['temp_min'].mean()
df_meteo = df_meteo.drop(df_meteo[df_meteo['temp_min'] > (mean_tmin + 3*std_tmin)].index)
df_meteo = df_meteo.drop(df_meteo[df_meteo['temp_min'] > (mean_tmin + 3*std_tmin)].index)

std_wind = df_meteo['wind'].std()
mean_wind = df_meteo['wind'].mean()
df_meteo = df_meteo.drop(df_meteo[df_meteo['wind'] > (mean_wind + 3*std_wind)].index)
df_meteo = df_meteo.drop(df_meteo[df_meteo['wind'] > (mean_wind + 3*std_wind)].index)

codes, uniques = pd.factorize(df_meteo['weather'])
df_meteo['weather_fact'] = np.array(codes)

# MLPRegressor
X = pd.DataFrame()

X["precipitations"] = df_meteo['precipitation']
X["temp_max"] = df_meteo['temp_max']
X["temp_min"] = df_meteo['temp_min']
X["wind"] = df_meteo['wind']

y = pd.DataFrame()
y['output'] = df_meteo['weather_fact']

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = MLPRegressor(random_state=0)

model.fit(X_train, y_train)


pickle.dump(model, open('model_meteo.pkl','wb'))