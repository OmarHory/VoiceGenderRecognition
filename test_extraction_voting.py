#!/usr/bin/env python

import os
import librosa
import librosa.display
import skimage
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parselmouth.praat import run_file
from joblib import load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
   

standard_scalar=load('./Models/std_scaler.bin') #Load the standard scalar that was fitted on the training-set

def run_praat_file(m, p, r):
    
    """
    A fuction used to return features from the given audio file using .praat file proposed by researchers:
    Nivja DeJong and Ton Wempe, Paul Boersma and David Weenink , Carlo Gussenhoven.     
    ...

    Parameters
    ----------
    m : str
        Path to file.
    p : str
        Path to dataset folder.
    r : str
        Path with .praat file exists.
        
    Returns
    ----------
    Object
        objects outputed by the praat script
        
    """
    sound=m
    path=p
    sourcerun=r
    
    assert os.path.isfile(sound), "Wrong path to audio file"
    assert os.path.isfile(sourcerun), "Wrong path to praat script"
    assert os.path.isdir(path), "Wrong path to audio files"

    try:
        objects= run_file(sourcerun, -20, 2, 0.3, "yes", sound, path, 80, 400, 0.01, capture_output=True)
        z1=str( objects[1]) # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
        z2=z1.strip().split()
        return z2
    except:
         return -2


def all_features_in_audio(m, p, r):
    """
    A fuction used to return features run_praat_file(m, p, r) function.
    ...

    Parameters
    ----------
    m : str
        Path to file.
    p : str
        Path to dataset folder.
    r : str
        Path with .praat file exists.
        
    Returns
    ----------
    Object
        features: Rate of speech, Count of Syllables, Count of Pause, Speaking time, Articulation, mean frequency, 
        minimum frequency, maximum freaquency.
        
    """
    try:
        
        z2 = run_praat_file(m, p, r)
        return float(z2[2]), float(z2[0]), float(z2[1]), float(z2[4]), float(z2[3]), float(z2[7]), float(z2[10]), float(z2[11])
    except:
        return -1,-1,-1,-1,-1,-1,-1,-1 #To raise an exception.


def extract_features(sound, sample_rate, name, path, path_praat_file):
    """
    A fuction used to extract features from the given audio file using librosa: 
    Publication reference: https://dawenl.github.io/publications/McFee15-librosa.pdf
    ...

    Parameters
    ----------
    sound : numpy.ndarray
        Signal wave-form extracted from librosa.
    sample_rate : numpy.ndarray
        Sampling rate for the generated signal from librosa.
    name : str
        Name of the audio file.
    path : str
        Path of audio file's directory..
    path_praat_file : str
        Path to .praat file.
        
    Returns
    ----------
    list
        Lists of extracted features from one single audio file.
        
    """
    
    # Extract MFCCs.
    # Capture timbral/textural aspects of sound. e.g: distinguishing people in speaking.
    # Frequency Domain Feature (By FT).
    # Approximate human auditory system (Models the way which humans interpret audio).
    # The coefficients can be from 13 to 40 (I have chosen 40).
    # Calculates MFCCs on each frame.
    # outputs MFCCs with Time.
    mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40).T, axis=0)
    
    # Short time Fourier transform to be as a given parameter in the chroma_stft feature extraction.
    # Computes several fft at different intervals.
    # Preserves time information.
    # Fixed frame size e.g: 2048 (The interval which FT will occur, then project it into the spectrogram).
    # Gives a spectrogram (time + fequency + magnitude).
    stft = np.abs(librosa.stft(sound))
    #librosa.display.specshow(librosa.amplitude_to_db(stft),sr=sample_rate)

    # Chromagram or intensity spectrogram.
    # A representation's pitch content within the time window is spread over the twelve different pitch classes.
    # Source: https://en.wikipedia.org/wiki/Chroma_feature
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    

    # mel-scaled spectrogram.
    # It is a Spectrogram with the Mel Scale as its y axis.
    # Which is the result of some non-linear transformation of the frequency scale. 
    # This Mel Scale is constructed such that sounds of equal distance from each other on the Mel Scale
    # Source: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
    mel = np.mean(librosa.feature.melspectrogram(sound, sr=sample_rate).T,axis=0)

    
    # Spectral Contrast considers the spectral peak, spectral valley and their difference in each sub-band.
    # Source: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.583.7201&rep=rep1&type=pdf
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    
    # With the harmonic relationship, tonnetz is a graphical representation of the tonal space.
    # To distinguish tones for an audio wave with it being a great feature for Gender Classification.
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound),sr=sample_rate).T, axis=0)
    #librosa.display.specshow(tonnetz,sr=sample_rate)
    
    #Retreive all features for an audio file.
    _, _, _, _, _, freq_mean, _, _ = all_features_in_audio(path, name, path_praat_file)
    
    
    return mfccs, chroma, mel, contrast, tonnetz, _, _, _, _, _, freq_mean, _, _



def model_preds_nn(modelNN, modelML, dataPath, filename, path_praat_file):
    """
    A function that crawls in the given file path to all audio file then extract features from them to create a labeled dataset
    with the extracted features and labels for Gender Classification.

    ...

    Parameters
    ----------
    modelNN : Depends on user input (KERAS OBJECT)
        DL Model.
    modelML : Depends on user input (sklearn OBJECT)
        ML Model.
    dataPath : str
        The path of the audio file.
    path_praat_file : str
        Path to .praat file.
    
        
    Returns
    -------
    pred
        Final prediction
        
    """
    
    #Initial Dataframe for the features.
    df = pd.DataFrame(columns=['mfccs','chroma','mel','contrast','tonnetz','freq_mean'])
 
    #Retreive the signal and the default sample_rate from librosa.
    sound, sample_rate = librosa.load(dataPath + '/' + filename, res_type='kaiser_fast')
                
    #Features are returned via extract_features.
    mfccs, chroma, mel, contrast, tonnetz, _, _, _, _, _, freq_mean, _, _ = extract_features(sound, sample_rate, dataPath, dataPath + '/' + filename, path_praat_file)
        
    #Add them to the dataframe.
    df.loc[0,'mfccs'] = mfccs
    df.loc[0,'chroma'] = chroma
    df.loc[0,'mel'] = mel
    df.loc[0,'contrast'] = contrast
    df.loc[0,'tonnetz'] = tonnetz
    df.loc[0,'freq_mean'] = freq_mean
    
    print("##########################################################")
    print('Finished Extracting Features..')
    if freq_mean == -1:
        return (-1,-1)

    df['freq_mean'][0] = np.array([df['freq_mean'][0]])
    features = np.array(df)
    _features = []
    _features.append(np.concatenate((features[0][0], features[0][1], 
                    features[0][2], features[0][3],
                    features[0][4], features[0][5]), axis=0))
    
    feature = np.array(_features)
    
    feature = standard_scalar.transform(feature)
    
    
    # Remove all Text.Grid files that were added in order to retreive the features, we do not need them anymore.
    #removeFilesByMatchingPattern('./Data','*.TextGrid')
    
    prednn = np.argmax(modelNN.predict(feature), axis=-1)
    predml = modelML.predict(feature)
    print('\nPredecting...\n')
    
    predml = np.where(predml == 0, 'Female', 'Male')
    prednn = np.where(prednn == 0, 'Female', 'Male')
    
    
    
    return (predml,prednn)



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

        