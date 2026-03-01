from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

#import ridge and standard scaler
model = pickle.load(open('models/ridge.pkl',"rb"))
scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form.get('temp'))
        RH = float(request.form.get('rh'))
        Ws = float(request.form.get('ws'))
        Rain = float(request.form.get('rain'))
        FFMC = float(request.form.get('ffmc'))
        DMC = float(request.form.get('dmc'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = model.predict(new_data_scaled)
        #after scaling and prediction we will show the result in home.html only
        return render_template('home.html', results = result[0])
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
