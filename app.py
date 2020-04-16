import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.converters import pdf_converter

df = pdf_converter(directory_path='./data/pdf')
# print(df)

cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib', max_df=1.0)

# Fit Retriever to documents
cdqa_pipeline.fit_retriever(df=df)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    query = request.form['query']
    query = str(query)
    prediction = cdqa_pipeline.predict(query) 
    output = prediction[0]
    
    return render_template('index.html', prediction_text= 'Answer: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)