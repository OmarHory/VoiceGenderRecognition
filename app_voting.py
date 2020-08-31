from flask import Flask, request, flash, jsonify
from tensorflow.keras.models import load_model
import test_extraction_voting as t
from werkzeug.utils import secure_filename
import pickle
import os

ALLOWED_EXTENSIONS = {'flac', 'wav'} #lossless formats/extensions.
MODELS_PATH = './models'
dataPath_test = "./temp" # Path of train speakers folders.
delimeter = '/' 
path_praat = './myspsolution.praat' # Path to .praat file.


# The initiation of the flask app.
app= Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#Neural Network Model 96.0%
modelNN = load_model(MODELS_PATH + '/latestNN')

#Convolutional Neural Network Model 96.6%
modelCNN = load_model(MODELS_PATH + '/latestCNN')

#Machine Learning Model (Random Forest) 94.74%
file = open(MODELS_PATH + '/latestML', 'rb')
modelML = pickle.load(file)
file.close()

#Check for allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({'output' : 'Invalid file.', 'success' : False}), 422 #Message with status code.
        file = request.files['file']
        print()
        if (file.filename).split(' ') != [file.filename]:
            flash('Rename Your file to be without spaces')
            return jsonify({'output' : 'No spaces in file name allowed.', 'success' : False}), 422 #Message with status code.
        if file.filename == '':
            flash('No selected file')
            return jsonify({'output' : 'Please select a file.', 'success' : False}), 422 #Message with status code.
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.isdir(dataPath_test):
                os.mkdir(dataPath_test)
            file.save(os.path.join(dataPath_test + delimeter, filename))
            print('File uploaded successfully.')
            pred, predcnn = test_prediction(file.filename)
            os.remove(dataPath_test+delimeter+file.filename+'.jpg')
            os.remove(dataPath_test+delimeter+file.filename)
            os.remove(dataPath_test+delimeter+file.filename.split('.')[-2] +'.TextGrid')
            
            if pred[0] == -1:
                prednn = 'Audio has not been recognized.'
                predml = 'Audio has not been recognized.'
            else:    
                predml = 'Male' if pred[0] == 'Male' else 'Female'
                prednn = 'Male' if pred[1] == 'Male' else 'Female'
            
            
            print('\nML:', predml,'\nNN:', prednn, '\nCNN:', predcnn, '\n')
            
            if prednn == 'Audio has not been recognized.':
                return jsonify({'output' : 'Audio has not been recognized.', 'success': True}), 200
            count_of_m=0
            count_of_f=0
            
            count_of_m, count_of_f = count_predection(prednn, count_of_m, count_of_f)
            count_of_m, count_of_f = count_predection(predml, count_of_m, count_of_f)
            count_of_m, count_of_f = count_predection(predcnn, count_of_m, count_of_f)
            
            # I called the keys as 'output' and 'success' so the API does not expose any information about the models.
            return jsonify({'output' : 'Male' if count_of_m > count_of_f else 'Female','success': True}), 200
        else:
            print('File extension is not allowed, use .flac or .wav')
            return jsonify({'output' : 'File extension is not allowed, use .flac or .wav', 'success' : False}), 422
        

def count_predection(predection, count_of_m, count_of_f):
    if predection == "Male":
       count_of_m+=1
    if predection == "Female":
       count_of_f+=1
    return count_of_m, count_of_f
       
def test_prediction(filename):
    return t.model_preds_nn(modelNN,modelML, dataPath_test, filename, path_praat), t.model_preds_cnn(modelCNN, dataPath_test, filename)
            
app.secret_key = os.urandom(24)
if __name__=="__main__":
    app.run()
    #app.debug = True
