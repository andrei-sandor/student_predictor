import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_math_grade = pickle.load(open('model_school_math_grade.pkl', 'rb'))
model_math_pass_fail = pickle.load(open('model_school_math_pass_fail.pkl', 'rb'))
model_reading_grade = pickle.load(open('model_school_reading_grade.pkl', 'rb'))
model_reading_pass_fail = pickle.load(open('model_school_reading_pass_fail.pkl', 'rb'))
model_writing_grade = pickle.load(open('model_school_writing_grade.pkl', 'rb'))
model_writing_pass_fail = pickle.load(open('model_school_writing_pass_fail.pkl', 'rb'))

model_meteo = pickle.load(open('model_meteo.pkl', 'rb'))

model_covid_none = pickle.load(open('model_covid_none.pkl', 'rb'))
model_covid_mild = pickle.load(open('model_covid_mild.pkl', 'rb'))
model_covid_moderate = pickle.load(open('model_covid_moderate.pkl', 'rb'))
model_covid_severe = pickle.load(open('model_covid_severe.pkl', 'rb'))

model_imdb = pickle.load(open('model_imdb.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    
     # Prediction Weather    
    features_weather = [float((request.form.get("precipitations"))), float((request.form.get("temp_max"))),
                     float((request.form.get("temp_min"))), float((request.form.get("wind")))]
    
    final_features_weather = [np.array(features_weather)]
    
    prediction_weather = model_meteo.predict(final_features_weather)[0]
    
    result_weather = ""
    
    if (round(prediction_weather, 0) == 0):
        result_weather = "drizzle"
    if (round(prediction_weather, 0) == 1):
        result_weather = "rain"
    if (round(prediction_weather, 0) == 2):
        result_weather = "sun"
    if (round(prediction_weather,0) == 3):
        result_weather = "snow"
    if (round(prediction_weather,0) == 4):
        result_weather = "fog"
    
    prediction_weather_html = "Predicted Weather is " + result_weather
    
    # Prediction IMDB    
    features_imdb = [int((request.form.get("Released_Year"))), int((request.form.get("Meta_score"))),
                     int((request.form.get("No_of_Votes"))), int((request.form.get("Gross")))]
    
    final_features_imdb = [np.array(features_imdb)]
    
    prediction_imdb = round(model_imdb.predict(final_features_imdb)[0],1)
    
    prediction_imdb_html = "Predicted IMDB is " + str(prediction_imdb)
    
    return render_template('index.html', prediction_weather = prediction_weather_html, 
                                        prediction_imdb=prediction_imdb_html)

if __name__ == "__main__":
    app.run(port=5000)