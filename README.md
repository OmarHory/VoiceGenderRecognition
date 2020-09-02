
# Voice Gender Recognition

[![N|Solid](https://hackernoon.com/hn-images/1*ChocH_eUxil5eaeXIsd3rw.png)](https://nodesource.com/products/nsolid)
# Why speech?
We are all acknowledged with the relevance of speech and its applications in our surroundings. Our phones are welcoming your voice more than ever. The use of voice-to-assistance concept is all around us. You ask your personal phone about today's weather right after you wake up by calling the name of that virtual-assistance inside your phone; some call it magic, others call it otherwise.
In those sets of notebook, present an intuitive application for _Gender classification_; where it receives an audio file and processes that into a single output; a 'male' or a 'female' voice. 
This [Kaggle Competition](https://www.kaggle.com/primaryobjects/voicegender) has a good-all understanding of the core basis of Gender Classification from different perspectives.
Gender classification is a decent problem for someone who is at the door-step of learning in this tremendous field. Regardless of all the work that must be done prior to building your model; but most Machine Learning/ Deep Learning practitioners come together toward the impact of what Voice-Recognition can ease the life of others.

# Dataset
There are a ton of well known [datasets](https://towardsdatascience.com/a-data-lakes-worth-of-audio-datasets-b45b88cd4ad) for speech recognition; one of the very famous is [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) with its two versions. In this notebook, [OPENSLR12](http://www.openslr.org/12/) is used for Gender Classification in the English language. This dataset is concerned with books as different speakers from different genders read different books on diverse periods.
This dataset is solely audio files with '.flac' formats. When you open the directory of the dataset you will see the following folder structure:

[![N|Solid](https://i.ibb.co/bsb1ksF/folder-structure.png)](https://nodesource.com/products/nsolid)

- **train-clean-100:** is a folder which contains multiple folders names with the _ID of the speaker_, inside each of folder there are multiple folders of the _ID of the book_; inside that the **.flac** audio files exist alongside a transcript **.txt** file.
- **BOOKS.TXT**: is a **.txt** file that contains the names of the books. There are 1568 books that were read by both genders.
- **CHAPTERS.TXT**: is a **.txt** file that contains the read chapters from the speakers.
- **SPEAKERS.TXT**: is a **.txt** file that contains the names of the speakers alongside their Gender; which will help us in determining the gender its corresponding audio file.
- The length of this training-set is 300 hours with 6.3 GB in size.
- [Training-set Download link ](http://www.openslr.org/resources/12/train-clean-100.tar.gz)
- The size of this Test-set (which comes from a different distribution than the training-set) is 346 MB.
- [Test-set Download link ](http://www.openslr.org/resources/12/test-clean.tar.gz)

# Instructions
The order of notebooks to run is as follows (to prevent any path conflicts):
- web-scraper-speech
- feature_extraction_nn_ml
- exploratory_data_analysis
- model_nn_and_ml
- cnn_modeling_and_feature_extraction

Running such project like this one requires some guidelines in order to work with convenience.
The structure of the files is as the order of this github repository.
The structure is extremely essential in order to run the notebooks in an efficient way without interruptions or path issues.
There are two options to run the project:
- **Google Colab**:

	If you use Google Colab, usually most Google Drive paths start with the following: **/content/drive/My Drive**
	If that is the case for you; add another folder inside called 'Google Colab' (With spaces). That is because I have done that on my local machine while training the DL/ML models; I recommend you do the same set of paths because everything is automated in the notebook; it already recognizes whether you are on Google Colab or on  Local Machine and will allocate the right paths for you. 
	_If you wish to have your own paths, feel free to change them; as I have left comments on each path to know exactly where that belongs._
- **Local Machine**:

	This one is the easiest and most convenient; every path is relative to the executed piece of code. I would suggest **Extracting the features** on your local machine; and running the models on Google Colab.

You will need the following libraries:
- Tensorflow 2.3.0 (To be compatible with Google Colab; if you want to run everything on your local machine then you do not need 2.3.0; if you have any issues running it; make sure to contact me as it has took me a lot of time to fix this compatibility) 
Download: **pip install tensorflow==2.3.0**
- keras-tuner [pip Link](https://pypi.org/project/keras-tuner/)
- Keras-Preprocessing [pip Link](https://pypi.org/project/Keras-Preprocessing/)
- Sklearn
- Librosa
- numpy
- pandas
- matplotlib
- Seaborn
- parselmouth [pip Link](https://pypi.org/project/praat-parselmouth/)
- soundfile
- joblib
	


# Crawler (Web-scraper)
In this notebook, a direct crawler is being used to pull both Training-set and Test-set to your local-directory and extract them then delete the .tar.gz file to retain space on your personal computer.
'BeautifulSoup' with 'requests' libraries were used to pull the data and extract. Thankfully, the OPENSLR website doesn't need any sort of form-filling or bypassing a CAPTCHA, so this job is quite easy to directly pull the required file.

# 'Brief' step-by-step description
## Pre-processing
_**NOTE: The pre-processing has been run on a local machine. If you do not want to run into path issues, follow the structure of this repository for the path structure in order to eliminate such issues.**_
-  **Walk-through your directory to your audio-files** *('.flac')*
    - That is done through the input of the **dataPath** in _dataset_creation_ function interacting with your current directory. 
     - Look for **.flac** audio files and assign them as inputs to _extract_features_ to retrieve features from this audio file. The features that were used are _MFCCs_, _Intensity (chroma)_, _mel-scaled_ _spectrogram_, _Spectral Contrast, tonnetz_. Also use the use of **parselmouth** library on python; it parses the audio file and uses a **.praat** file in order to extract other features like the #of syllables in a speaking period, speaking time, articulation, count of pauses, mean-frequency, min-frequency, max-frequency.
    - Add the features to your DataFrame or Dataset.
     - Extract the gender (target-label) of the audio file and add it to another DataFrame with the use of **SPEAKER.TXT**.
      - Join both DataFrames that result in the full dataset after you pass through all audio-files.
      - Output the dataset as a **.csv**.
      - Save the features vector and pickle it.
- **Load features vector.**
	- Shuffle training and test-set, the previous pre-processing step had the speakers lined up with respect to their audio-file.
     - Split to X and Y for both the training-set and Test-set.
     - **(Optional):** Use StratifiedKFold on training-set to ensure the data is balanced. I didn't use it on my training-set because I do not need a test-set that is from the same distribution as my training-set, not the type of problem that requires this.
     - Create a _dev/cross-validation set_ from the _test-set_ because both of these sets must be from the same distribution and different from the _training-set._
    - One-hot-encode the labels to be able to use it in the softmax activation function in the output neuron rather than having a restricted '0' or '1'; to have a conditional probability instead.
    - Use standardization on the feature vector to have a zero-mean with a standard deviation.
## Exploratory Data Analysis (EDA)
In this section, an EDA is performed to perceive the various types of features that are handled during the Machine learning/ Deep Learning model. It has gone through a diverse collection of features that are extracted once the audio-file is passed to the _extract_feature_ function. Those features are:
-   MFCCs
-   Chroma (intensity)
-   mel-scaled spectrogram
-   Spectral Contrast
-   Tonnetz
-   Audio Duration
-   Rate of speech
-  \# of syllables
-   Speaking time (s)
-   Articulation (Speed)  [Reference (Publication)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2790192/)
-   Count of pauses with fillers
-   Mean frequency
-   Minimum frequency
-   Maximum frequency

The above features are helpful in nature to visualize their properties and get a glimpse of the behavior of this data. EDA is needed the most in the Speech Recognition field, because the features are so 'dependent', removing one feature or adding an 'unnecessary' feature might outcome unwanted results. Being extremely selective with the features is mandatory to ensure validity between different models.
Intuitively, there are features that are not useful when it comes to modeling; such as: 
- Audio Duration
- \# of syllables
- Speaking Time
- Count of pauses with fillers.

The reason behind that is those features; if they are given to you for example, you will not be able to specify whether the speaker is a male of female.
_In summary_, the above features are made for visualization, to understand how your data is centered in a certain distribution.
The rest of the features are selected with an approach, but the final features selected for modeling were the following:

- MFCCs
-   Chroma (intensity)
-   mel-scaled spectrogram
-   Spectral Contrast
-   Tonnetz
-  Mean Frequency

The reason behind that depends solely on the nature of the data, the visualizations that are available in the notebook do explain this in details.

## Modeling
_**NOTE: Training the models was done on Colab-Pro as Google offers a great runtime-environment with a GPU & TPU to use for an affordable price.**_
- **Artificial Neural Network:**
     - Baseline Neural Network
     - Improved Neural Network
     - Hyper-parameter tuned deep Neural Network model
 - **Convolutional Neural Network:**
     - Baseline Convolutional Neural Network
     - Improved Convolutional Neural Network
     - Hyper-parameter tuned deep Convolutional  Neural Network 
 - **Machine Learning Models:**
	 - Logistic Regression
	 - KNN
	- Stochastic Gradient Descent
	- Random Forest
	- XGBOOST

## Hyper-parameter tuning
- **Artificial Neural Network:**
	- Create a baseline model to compare its loss and accuracy to other models.
	- Create an improved model upon the baseline and compare its accuracy and loss.
	- Use Keras Tuner with the same number of parameters as the improved model with different range of values to random-search with a given number of trials and return the best model.
- **Convolutional Neural Network**
	- Create a baseline model to compare its loss and accuracy to the improved model.
	- Create an improved model upon the baseline and demonstrate its accuracy.
	- Use Keras Tuner with the some different parameters and ranges to random-search and return the best model.
- **Machine Learning Models:**
	As any ML model, keeping the parameters on default **might** get the job done, since Machine learning models are way simpler than NN; we will perform a Bruteforce/ GridSearch on some models.
	- Run GridSearchCV from sciket-learn on Logistic Regression, SGD, then return the best set of parameters and fit again to plot necessary visualizations.
	- For KNN, we will perform a simple K-Value optimizer to find the best K-Value for the model. Afterwards, we will fit the model again to plot necessary visualization of accuracy metrics.
	- In terms for the Random Forest and XGBOOST, their GridSearch is extremely computationally expensive and even on Google Colab it was taking 10+ hours, selectively choosing the parameters would do the job.
	- Demonstrate a comparison in terms of Accuracy on the test-set and the Area under Curve (AUC).

## Deploy
This project will be served as a Restful API with a docker-file to build to be tested on for your convenience.
Flask is used to allow testing for your own voice; keeping in mind that the available audio-file-extensions are: **'flac** and **.wav**; that is due to the lossless nature of those extensions; especially the .wav; it is *LINEAR16*  or LINEAR PCM; which is an example of uncompressed audio in that the digital data is stored exactly as the standards imply.
You can run the Flask API on your local machine and use *Postman* or *Advanced Restful Client* on Google Chrome App or any other Restful client that you prefer to test your voice on those models.

*There are two API's that are included in the repository, one with the best generated model and the other one is based on a voting-behavior system, that takes into account the majority of decisions.*

The procedure of the prediction for the **best-generated-model** (CNN) goes as follows:
- Upload a .flac or .wav audio-file to the server, it gets saved in _./Data/temp_.
- Feature Extraction for CNN.
- Delete audio-file and spectrogram to retain space.
- Predict.

The procedure of the prediction for the **voting-behavior** from the best model of ANN, CNN and ML, then return the prediction with the most votes (Note: ANN and ML only can predict the unrecognizability of the audio-file, that is due to the *.praat* file from parselmouth, which is not available for CNN as it is only concerned with images.

The process of predicting the gender goes as follows:
- Upload a .flac or .wav audio-file to the server, it gets saved in _./Data/temp_.
- Feature extraction whether for CNN or ANN.
- Delete audio-file and spectrogram to retain space.
- Predict.
- Most votes is the determined final prediction.

Personally, I do prefer the first version which is the **best-generated-model**. This eliminates confusion between models, there are mainly four reasons for this:
- CNN is the only model that uses images as features while ANN and ML use the same numeric features, so they will be biased toward each other when it comes to learning and predicting.
- There is a feature of ANN and ML model is that it can detect *unnatural sounds* and that is not available in CNN for the reason that we do not have a third label that identifies *'Unrecognized voice'* but in ANN and ML this feature is doable through feature extraction by using parselmouth library with the .praat file (yet it needs improvements in this regard).
- CNN can generalize more because there is no a 'clear' feature extraction, what we actually do is to extract a spectrogram image and can obtain features through the CNN Architecture. Also, CNN offers the availability of *Image Augmentation* which can increase the generalization of the model; as I have noticed this on multiple voices that I have personally recorded with noise and bad microphone quality yet it still recognizes the gender.
- ANN has gotten used to the train and validation data regarding reading from a textbook. ANN sometimes misinterpret the reader if the reader is: singing, speaking fast or young/very old in age.

## Future Work
For further inspection, I would love to try out different methodologies like:
- Another approach would be to use **Low-pass filter** to do noise-reduction; especially the background noise, that would be done before the pre-processing step, then on test-time; we can pass those audio files to the Low-pass filter before feeding them to the model.
- A possible approach is to create a separate model on a different dataset; then blending the models altogether which will result in a more efficient Gender Classifier that handles many possible set of circumstances.
- Train models on the 500 hour version and to observe where the model architecture is compared to the other ones.
- Do feature extraction in a different method, try out different combinations to improve the generalization of the models.
- Add a third label of 'Unrecognized Voice', by making sure the audio-file imported is a person speaking.

## Last words
_As the first Voice-recognition challenge, I am quite satisfied with all the stuff that I have learned in voice and signal processing, I have a lot in mind. Consequently, I will keep improving on this repository in terms of modeling and feature selection that I truly conceive that the former needs some tweaking.
I do genuinely believe that this project has a potential to become big; if not my biggest project of all time. I really want to thank Google Colaboratory for this flexible environment, I've been using it for a year now and I have never complained.
Also I would like to thank OPENSLR for this fulfilling dataset, also with the availability to edit and share anywhere on the web [License](https://creativecommons.org/licenses/by/4.0/).
Thanks for reading if you have reached until this final piece, my pleasure.
P.S.: If you record your voice and get mis-recognized in an opposite manner, I extremely apologize :D. If that happens, please, I would love your input for improving the generalization of the models.
Do not hesitate to contact me at anytime on my personal email if you have any inquires or advice that would have an impact on this repository : o_hawary@hotmail.com ._

## Useful Research Papers & Articles
- [The Difference Between a Male and Female Voice Over](https://matinee.co.uk/blog/difference-male-female-voice/)
- [Voice Gender Recognition Using Deep Learning](https://www.researchgate.net/publication/312219824_Voice_Gender_Recognition_Using_Deep_Learning)
- [Neural architectures for gender detection and speaker identification ](https://www.tandfonline.com/doi/full/10.1080/23311916.2020.1727168)
- [Automatic Identification of Gender from Speech](http://www.cs.columbia.edu/~sarahita/papers/speech_prosody16.pdf)
- [VOICE-BASED GENDER IDENTIFICATION IN MULTIMEDIA APPLICATIONS](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.143.7087&rep=rep1&type=pdf)
- [Gender Identification by Voice](http://cs229.stanford.edu/proj2014/Kunyu%20Chen,%20Gender%20Identification%20by%20Voice.pdf) 
- [Voice Gender Recognition Using Deep Learning](https://download.atlantis-press.com/article/25868884.pdf) 
