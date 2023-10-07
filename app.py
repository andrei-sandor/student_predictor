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

model_math_mid1 = pickle.load(open('model_math_mid1.pkl', 'rb'))
model_math_mid2 = pickle.load(open('model_math_mid2.pkl', 'rb'))
model_math_final= pickle.load(open('model_math_final.pkl', 'rb'))

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
    
    # Prediction Math
    features_math = []
    
    math_gender = request.form['gender_school']
    if (math_gender == "Male"):
        features_math.append(1)
    else:
        features_math.append(0)
        
        
    math_age = request.form['age']
    features_math.append(int(math_age))
    
    math_fam = request.form['fam_size']
    if (math_fam == "le3"):
        features_math.append(1)
    else:
        features_math.append(0)
        
    math_travel = request.form['travel']
    if (math_travel == "0-15min"):
        features_math.append(1)
    if (math_travel == "15-30min"):
        features_math.append(2)
    if (math_travel == "30-60min"):
        features_math.append(3)
    if (math_travel == ">60min"):
        features_math.append(4)
        
    math_study = request.form['study_time']
    if (math_study == "<2 hours "):
        features_math.append(1)
    if (math_study == "2-5 hours study"):
        features_math.append(2)
    if (math_study == "5-10 hours study"):
        features_math.append(3)
    if (math_study == ">10 hours study"):
        features_math.append(4)
    
    math_act = request.form['extra']
    if (math_act == "Yes"):
        features_math.append(1)
    else:
        features_math.append(0)
        
    math_free = request.form['free']
    if (math_free == "1"):
        features_math.append(1)
    if (math_free == "2"):
        features_math.append(2)
    if (math_free == "3"):
        features_math.append(3)
    if (math_free == "4"):
        features_math.append(4)
    if (math_free == "5"):
        features_math.append(5)
        
    math_friends = request.form['friends']
    if (math_friends == "1"):
        features_math.append(1)
    if (math_friends == "2"):
        features_math.append(2)
    if (math_friends == "3"):
        features_math.append(3)
    if (math_friends == "4"):
        features_math.append(4)
    if (math_friends == "5"):
        features_math.append(5)
    
    
    features_math.append(int((request.form.get("absences"))))
    
    final_features_math = [np.array(features_math)]
    
    prediction_math_mid1 = round(model_math_mid1.predict(final_features_math)[0]/20*100,0)
    
    final_prediction_math_mid1 = "Your predicted midterm 1 Math grade is " + str(prediction_math_mid1)

    
    
    prediction_math_mid2 = round(model_math_mid2.predict(final_features_math)[0]/20*100,0)
    
    final_prediction_math_mid2 = "Your predicted midterm 2 Math grade is " + str(prediction_math_mid2)
    
    prediction_math_final = round(model_math_final.predict(final_features_math)[0]/20*100,0)
    
    final_prediction_math_final = "Your predicted final Math grade is " + str(prediction_math_final)
    
    
    # Prediciton covid
    features_covid = []
    
    covid_fever = request.form['fever']
    if (covid_fever == "Yes Fever"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_tired = request.form['tired']
    if (covid_tired == "Yes tired"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_cough = request.form['cough']
    if (covid_cough == "Yes cough"):
        features_covid.append(1)
    else:
        features_covid.append(0)
    
    covid_breath = request.form['breath']
    if (covid_breath == "Yes breath"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_sore_throat = request.form['Sore-Throat']
    if (covid_sore_throat == "Yes Sore-Throat"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_none_symptoms = request.form['None_Symptom']
    if (covid_none_symptoms == "Yes None_Symptom"):
        features_covid.append(1)
    else:
        features_covid.append(0) 
            
    covid_pains = request.form['Pains']
    if (covid_pains == "Yes Pains"):
        features_covid.append(1)
    else:
        features_covid.append(0)    
        
    covid_nasal = request.form['Nasal-Congestion']
    if (covid_nasal == "Yes Nasal-Congestion"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_nose= request.form['Runny-Nose']
    if (covid_nose == "Yes Runny-Nose"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_diarrhea= request.form['Diarrhea']
    if (covid_diarrhea == "Yes Diarrhea"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_age = request.form['Age_20-24']
    if (covid_age == "Yes Age_20-24"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_female = request.form['Female']
    if (covid_female == "Yes Female"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_male = request.form['Male']
    if (covid_male == "Yes Male"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_transgender = request.form['Transgender']
    if (covid_transgender == "Yes Transgender"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_contact_idk = request.form['Contact_Dont-Know']
    if (covid_contact_idk == "Yes Contact_Dont-Know"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_contact_no = request.form['Contact_No']
    if (covid_contact_no == "Yes Contact_No"):
        features_covid.append(1)
    else:
        features_covid.append(0)
        
    covid_contact_yes = request.form['Contact_Yes']
    if (covid_contact_yes == "Yes Contact_Yes"):
        features_covid.append(1)
    else:
        features_covid.append(0)
    
    final_features_covid = [np.array(features_covid)]
    
    prediction_covid_none = model_covid_none.predict(final_features_covid)[0]
    
    final_prediction_covid_none = ""
    
    if prediction_covid_none == 0:
        final_prediction_covid_none = "No covid"
    else:
        final_prediction_covid_none = "Covid"
        
        
    prediction_covid_mild = model_covid_mild.predict(final_features_covid)[0]
    
    final_prediction_covid_mild = ""
    
    if prediction_covid_mild == 1:
        final_prediction_covid_mild = "Possibility of mild Covid"
    else:
        final_prediction_covid_mild = "No mild covid"    
    
    prediction_covid_moderate = model_covid_moderate.predict(final_features_covid)[0]
    
    final_prediction_covid_moderate = ""
    
    if prediction_covid_moderate == 1:
        final_prediction_covid_moderate = "Possibility of moderate Covid"
    else:
        final_prediction_covid_moderate = "No moderate covid"
        
    prediction_covid_severe =  model_covid_severe.predict(final_features_covid)[0]
    
    final_prediction_covid_severe= ""
    
    if prediction_covid_severe == 1:
        final_prediction_covid_severe = "Possibility of severe Covid"
    else:
        final_prediction_covid_severe = "No severe covid"
    
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
    
    return render_template('index.html', prediction_math_mid1 = final_prediction_math_mid1,
                                        prediction_math_mid2 = final_prediction_math_mid2,
                                        prediction_math_final = final_prediction_math_final,
                                        prediction_covid_none = final_prediction_covid_none,
                                        prediction_covid_mild = final_prediction_covid_mild,
                                        prediction_covid_moderate = final_prediction_covid_moderate,
                                        prediction_covid_severe = final_prediction_covid_severe,
                                        prediction_weather = prediction_weather_html, 
                                        prediction_imdb=prediction_imdb_html)

if __name__ == "__main__":
    app.run(port=5000)