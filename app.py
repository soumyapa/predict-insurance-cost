import sys
import flask
from flask import render_template , url_for,request , redirect
app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import pickle

with open('knn.pkl', 'rb') as picklefile:
    PREDICTOR = pickle.load(picklefile)


#-------- ROUTES GO HERE -----------#


# This method takes input via an HTML page
@app.route('/page')
def page():
   return render_template('page.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

        inputs = flask.request.form

        age = inputs['age'][0]
        children = inputs['children'][0]
        sex_male = inputs['sex_male'][0]
        smoker_yes = inputs['smoker_yes'][0]
        region_northwest = inputs['region_northwest'][0]
        bmi_Obese = inputs['bmi_Obese'][0]

        item = np.array([age , children , sex_male, smoker_yes , region_northwest, bmi_Obese]).reshape(-1,6)
        preds = PREDICTOR.predict(item)
        #return flask.jsonify(preds[0])
        return render_template('insurance_result.html', results = preds[0])


# A welcome message to test our server
@app.route('/')
def index():
	return redirect(url_for('page'))

if __name__ == '__main__':
    '''Connects to the server'''

    #HOST = '127.0.0.1'
    #PORT = 4000
    app.run()
