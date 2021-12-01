#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__,template_folder='template')
model = joblib.load("model.pkl")


# In[8]:


@app.route('/')
def home():
    return render_template('index.html')


# In[10]:


@app.route('/predict',methods=['POST','GET'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html',prediction_text='THE OVERALL SCORE IS {}'.format(output))

if __name__ == "__main__":
    app.run(debug = True)


# In[ ]:




