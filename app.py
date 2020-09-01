from flask import Flask, request, flash, jsonify
from tensorflow.keras.models import load_model
import test_extraction as t
from werkzeug.utils import secure_filename
import os

ALLOWED_EXTENSIONS = {'flac', 'wav'} #lossless formats/extensions.
MODELS_PATH = './models'
dataPath_test = "./temp" # Path of train speakers folders.
delimeter = '/' 


# The initiation of the flask app.
app= Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #16 MB Max Size.

#Convolutional Neural Network Model 96.6%
modelCNN = load_model(MODELS_PATH + '/latestCNN')


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
            flash('Rename Your file to be without spaces.')
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
            predcnn = test_prediction(file.filename)
            os.remove(dataPath_test+delimeter+file.filename+'.jpg') #Remove the image to retain space.
            os.remove(dataPath_test+delimeter+file.filename)
            
            print('\nPrediction: ',predcnn, '\n')
            

            # I called the keys as 'output' and 'success' so the API does not expose any information about the model.
            return jsonify({'output' : predcnn ,'success': True}), 200
        else:
            print('File extension is not allowed, use .flac or .wav')
            return jsonify({'output' : 'File extension is not allowed, use .flac or .wav', 'success' : False}), 422
            
            
            
def test_prediction(filename):
    return t.model_preds_cnn(modelCNN, dataPath_test, filename)
            
app.secret_key = os.urandom(24)
if __name__=="__main__":
    app.run()
    #app.debug = True
