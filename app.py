import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("boston_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    return jsonify(output[0])

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    new_data = scaler.transform(np.array(data).reshape(1, -1))  # Fixed this line
    output = model.predict(new_data)[0]
    return render_template("index.html", predicted_test=f"The final price is {output}")

if __name__ == "__main__":
    app.run(debug=True)
