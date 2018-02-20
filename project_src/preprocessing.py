from utile import *
import librosa
import os
import numpy as np

'''
Preprocess data using diferent features
'''

def mel_spec(y, sr):
    '''
    Compute the mel spectogram
    '''
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, hop_length = len(y)/20) # shape = (128, 21)
    log_S = librosa.power_to_db(S, ref=np.max) 
    return log_S

def mfcc_coeffs(y, sr):
    '''
    Compute the mfcc coeffs
    '''
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, hop_length = len(y)/20) # shape = (128, 21)
    log_S = librosa.power_to_db(S, ref=np.max)

    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13) # shape = (13, 21)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2) # shape = (13, 21)
    return delta2_mfcc

def preprocessing(filename, fct = mel_spec):
    '''
    Preprocess a file
        - filename: path to the file
        - fct: type of preprocessing (default : mel_spec)
    '''
    y, sr = librosa.load(filename)
    y = librosa.effects.trim(y, top_db=15)[0]
    return fct(y, sr)

def get_label(filename):
    return filename.split("/")[-2]

labels = {
	"yes":0,
	"no":1,
	"up":2,
	"down":3,
	"left":4,
	"right":5,
	"on":6,
	"off":7,
	"stop":8,
	"go":9,
    "other":10
}

def preprocessing_all(file_per_class = 500, folder = "../data/train/audio", labels_encoder = labels):
    '''
    Preprocess all data and save the data and labels as numpy arrays:
        - file_per_class: number of file preprocessed for each class (np.infty if you want to preprocess all)
        - folder: path to folder containg the data
        - labels_encoder: dic
    '''
    path = "../data/preprocessed"
    X = np.zeros((11*file_per_class, 128, 21))
    y = np.zeros((11*file_per_class, 11))
    if not os.path.exists(path):
        os.makedirs(path)
    cpt_labels = {}
    for key in labels_encoder.keys():
        cpt_labels[key] = 0
    g = generator()
    i = 0
    print(labels_encoder)
    while min([cpt_labels[key] for key in cpt_labels.keys()]) < file_per_class:
        filename = g.next()
        label = get_label(filename)
        if label not in labels_encoder.keys():
            label = "other"
        print(label)
        if int(cpt_labels[label]) < file_per_class:
            X[i,:,:] = preprocessing(filename)
            y[i,labels[label]] = 1
            cpt_labels[label] += 1
            i += 1
    np.save(os.path.join(path, "input.npy"), X)
    np.save(os.path.join(path, "output.npy"), y)
    return

if __name__ == "__main__":
    print("Start preprocessing")
    preprocessing_all()        
    

	
	
