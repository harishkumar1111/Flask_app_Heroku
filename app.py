
# This is basically the heart of my flask 


from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
import xgboost

app = Flask(__name__)  # intitialize the flaks app  # common 

#loading the sparse file which have processed features
# Raw File which have missing values , Outlier , ..

# You can python to process and creat final DF or object which then pass in Pickle

# loading the data 
# here i am loading npz , you can load csv , xlsx , databse conenection 
Xtest_Scenerio1_for_flask  = sparse.load_npz("dataset/Xtest_Scenerio1_for_flask.npz")
# here you can database connector 
# external API (Twitter API )

#Xtest_Scenerio1_for_flask  - holding my data whch will render on UI 

# http:baseurl/age_prediction



@app.route('/age_prediction/')
def age_pred():
    pipeline = pickle.load(open('pickle/xgboost_age_scenerio1.pkl', 'rb'))
    prediction = pipeline.predict_proba(Xtest_Scenerio1_for_flask)
    df_pred_prob = pd.DataFrame(prediction, columns=['0-24', '25-32', '32+'])
    df_pred_prob['max_prob'] = df_pred_prob[['0-24','25-32','32+']].max(axis=1)
    df_pred_prob['max_prob_class'] = df_pred_prob.idxmax(axis=1)
    pred_df = df_pred_prob.iloc[0:50,:]
    df_data = pd.read_csv('test_scenerio1_for_flask.csv')
    df_concat = pd.concat([pred_df,df_data.iloc[0:50,:]],axis = 1)
    df_concat['campaigns'] = np.where(df_concat.max_prob_class == '0-24', 'Campaign 4',np.where(df_concat.max_prob_class == '25-32', 'Campaign 5' , 'Campaign 6'))
    df_concat = df_concat[['device_id', 'max_prob','max_prob_class','campaigns']]
    return  render_template('view.html',tables=[df_concat.to_html(classes='age')], titles = ['NAN', 'Age Prediction'])






# Any HTML template in Flask App render_template

if __name__ == '__main__' :
    app.run(debug=True )  # this command will enable the run of your flask app or api
    
    #,host="0.0.0.0")






