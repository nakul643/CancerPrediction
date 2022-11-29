import numpy as np
import os
from flask import Flask, request, jsonify, render_template,redirect,url_for,send_from_directory,flash
import pickle
from werkzeug.utils import secure_filename
from datetime import datetime
from script import process_csv

ALLOWED_EXTENSIONS= set(['csv'])
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
# Create flask app

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
@app.route("/upload",methods=['GET','POST'])
def upload():
    if request.method=="POST":
        file=request.files['file']
        if file and allowed_file(file.filename):
            filename= secure_filename(file.filename) 
            # new_filename= f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
            save_location=os.path.join("uploadfiles",filename)
          
            file.save(save_location)
            output=  process_csv(save_location) 
        
            return render_template("download.html")
            
@app.route("/download")
def download():
    output="result.csv"
    save_file=f'result_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}.csv' 
    return send_from_directory('output',output,download_name=save_file)
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
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug=True
    
    app.run()
