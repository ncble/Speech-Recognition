from utile import *
import librosa
import os
import time
import numpy as np

'''
Cut silence in data -> unefficient because size_after > size_before...
'''

def reduce_audio_file(filename, new_path):
    '''
    Open a file, remove silence and save it to another location:
        - filename: path to the file
        - new_path: path to new file
    '''
    y, sr = librosa.load(filename, sr=22050)
    y = librosa.effects.trim(y, top_db=15)[0]
    librosa.output.write_wav(new_path, y, sr)

def reduce_all(folder, new_folder, max_iter = 100):
    '''
    Remove silence from all the file and save them to another location:
        - folder: path to the folder containing the files
        - new_folder: path to the new folder to save the files
        - max_iter: max number of files to reduce
    '''
    g = generator(root_dir=folder)
    new_folder_split = new_folder.split("/")
    i = 0
    for filename in g:
        file_split = filename.split("/")
        if file_split[-1][-3:] == "wav": # check that the file is a ".wav"
            new_path = new_folder +"/"+ "/".join(file_split[len(new_folder_split):-1])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            reduce_audio_file(filename, os.path.join(new_path, file_split[-1]))
        i += 1
        if i > max_iter:
            return

if __name__ == '__main__':
    print("Start test data")
    if not os.path.exists('./data_reduced/train/audio'):
        os.makedirs('./data_reduced/train/audio')
    reduce_all('./data/train/audio/', './data_reduced/train/audio', max_iter = np.infty)

    print("Start test data")
    if not os.path.exists('./data_reduced/test/audio'):
        os.makedirs('./data_reduced/test/audio')
    reduce_all('./data/test/audio/', './data_reduced/test/audio', max_iter = np.infty)
    
