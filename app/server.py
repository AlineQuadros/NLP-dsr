from flask import Flask, jsonify, request, render_template
from processing import process_data, load_precalc 
from joblib import dump, load
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    data = pd.read_csv("clean_dataset.csv")
    ids = np.random.randint(0, len(data), 3)
    abstracts = data.abstract[ids]
    abstracts.reset_index(drop=True, inplace=True)
    return render_template('index.html', abstract1=abstracts[0], abstract2=abstracts[1])

@app.route('/topics', methods=['GET', 'POST'])
def index():
    data = pd.read_csv("clean_dataset.csv")
    ids = np.random.randint(0, len(data), 3)
    abstracts = data.abstract[ids]
    abstracts.reset_index(drop=True, inplace=True)
    return render_template('topics.html', abstract1=abstracts[0], abstract2=abstracts[1])

@app.route('/choose', methods=['GET', 'POST'])
def load_similar():
    user = request.form['nm']
    return render_template('index.html', name=user)

if __name__ == "__main__":
    app.run(debug=True)
