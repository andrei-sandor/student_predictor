# predictor_student

Tools:

    Python (Pandas, Scikit-Learn)
    Machine Learning (Regression, Classification, Imbalanced Techniques, Train test split, classfication report, R2Score, metrics for regression)
    EDA
    Flask
    HTML and CSS
    Pickle
    Jupyter Notebook

Creadted alone a Flask web app. This app called student predictor is a fundamental tool for university students. The first thing that it can do is to predict the grade of a student in function of many parameters for a midterms and final exam. With the current situation, students want to know if they have Covid. I used a dataset from Kindle that checks many symptoms to predict if the student has Covid. Furthermore, a student wants to enjoy his/her free time, that's why I included a machine learning alogirthm to predict if a movie is good or not. Finally, students want to optimize their presence to school by knowing the weather. I created an algorithm that predicts the whether in function of temperature, precipitations and wind.

To create this app, this is what I have done. First of all, I used datasets from Kindle about exams, Covid, wheather and IMDB of movies. Then, I used to csv files inside a Jupyter Notebook. Then, I cleaned the datasets by doing EDA, dealing with outliers and factorizing. Then, I investigate the machine learning algorithm. I investigate regression algorithms (Linear Regression, MLPRegressor, DecisionTreeRegressor, RandomForestRegressor and SVC. Then, I used r2_score, MAE and MSE to determine the best algorithm by doing 80/20. Then, I also investigated classifiaction algorithms (GaussianNB, KNN and DecisionTreeClassfier) by using a classification report. Then, I kept everything up to and including the fit inside a Python file that will generate a pickle file. Then, I started the Flask part. I first created the frontend by using HTML (Dropdown lists,inputs and titles) and the CSS for colors. Then, I linked the frontend to backend with Flask. I used the inputs of the frontend by linking them to the backend by using the pickle files that serialize the Machine Learning model.

# Use the app

Go to the main folder and run the command 

- python app.py

Go to the localhost mentioned and input some values. After, click predict and you can see the predictions!
