import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    print(prediction)
    if prediction=="M":
        prediction="Malignant"
    elif prediction=="B":
        prediction="Benign"
    return render_template("index.html", prediction_text = "The cancer status is {}".format(prediction))

if __name__ == "__main__":
    app.debug=True
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.run()
