#!/usr/bin/env python

import librosa
import librosa.display
import skimage
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def model_preds_cnn(model, path, filename, delimeter = '/'):
    """
    A function that saves the figure of the spectrogram to a destination folder.
    ...

    Parameters
    ----------
    model : Depends on user input (KERAS OBJECT)
        DL Model.
    path : str
        The directory of the audio file.
    filename : str
        Path of audio file.
    delimeter : str
        Separator.
        
    Returns
    -------
    str
        Final prediction
        
    """
    sound, sample_rate = librosa.load(path + delimeter + filename, res_type='kaiser_fast', sr=None)
    #To remove axis and lables to keep the image as it without any additional framing.
    fig = plt.figure(figsize=[1,1])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    # The melspectrogram it can be considered as the visual representation of a signal.
    # In addition, it represents how the spectrum of frequencies vary over time.
    # This helps a lot, especially when the specification of the minimum frequency and the maximum frequency is present!
    spectrogram = librosa.feature.melspectrogram(y=sound, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), fmin=50, fmax=280, x_axis='time', y_axis='mel')
    
    file_name  = path + delimeter + filename + '.jpg'
    plt.savefig(file_name, dpi=500, pad_inches=0, bbox_inches='tight') # Those 

    plt.close()
    
    img = skimage.io.imread(path + delimeter + filename + '.jpg')         
    img = skimage.transform.resize(img, (64, 64, 3))  
    img = img[np.newaxis, ...]         
    prediction = model.predict(img)     

    if prediction[0][0] > prediction[0][1]:
        return 'Female'
    else:
        return 'Male'

        