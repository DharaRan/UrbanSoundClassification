import glob
import os
import librosa
import numpy as np
import pandas as pd
from IPython import get_ipython
import pickle
from sklearn.preprocessing import normalize, MinMaxScaler

"""
This code extracts the features of the raw sound files and shape into the 
desired shape.
"""

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels =10 #len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def get_random_patch(frames, num_frames):
    # TODO: Randomize
    start_frame = 0
    return frames[:, start_frame:start_frame + num_frames]


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features_spectrograms(parent_dir, sub_dirs, file_ext="*.wav", bands=128,
                                  frames=128, normalize_data=False):  # sliding window spectrals
    log_specgrams = []
    labels = []
    window_size = 512 * (frames - 1)
    #print(window_size)
    
    if normalize_data:
        rescale = MinMaxScaler(feature_range=(0, 1), copy=True)  # rescale between 0 and 1

    for l, sub_dir in enumerate(sub_dirs):
        print('parsing %s...' % sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print('fn is ', fn)
            sound_clip, s = librosa.load(fn)
            label = fn.split('/')[fn.count('/')].split('-')[1]  # TODO change to fn.split('/').pop().split('-')[1]
            #print(len(sound_clip))
            for (start,end) in windows(sound_clip,window_size):
                #print("Third for loop")
                start=int(start)
                end=int(end)
                if(len(sound_clip[start:end]) == window_size):
                    print(start,end)
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                    logspec = librosa.logamplitude(melspec)
                    if normalize_data:
                        logspec = rescale.fit_transform(get_random_patch(logspec, frames).T)

                    log_specgrams.append(logspec)
                    labels.append(label)
            
   
    log_specgrams = np.array(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    labels = one_hot_encode(np.array(labels, dtype=np.int))
    # # Add channel of deltas
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    return features, labels


def extract_features_spectrograms_norm(parent_dir, sub_dirs, file_ext="*.wav", bands=128,
                                       frames=128, normalize_data=True):  # sliding window spectrals
    # helper for dump_features
    return extract_features_spectrograms(**locals()) # pass all input vars

def extract_features_means(parent_dir, sub_dirs, file_ext='*.wav'):
    features, labels = np.empty((0, 193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        print('parsing %s...' % sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:  # sometimes failss??
                # mean value of spectral content for Feed Forward Net
                X, sample_rate = librosa.load(fn)
                stft = np.abs(librosa.stft(X))
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
                contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            except:
                print('error, skipping...', fn)
                pass
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split('/')[fn.count('/')].split('-')[1])
    return np.array(features), one_hot_encode(np.array(labels, dtype=np.int))




AUDIO_DIR =  'C:\\Dhara\\Cs698_DataScienceTopics\\Project\\UrbanSound8K\\audio\\'
SAVE_DIR = 'C:\\Dhara\\Cs698_DataScienceTopics\\Project\\UrbanSound8K\\data139features\\'
TEST = ['fold10','fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9']


def dump_data(path, objects, names, ext='.pkl'):
    for i, o in enumerate(objects):
        print('dumping %s...' % names[i])
        pklfile=open(path + names[i] + ext, "wb")
        pickle.dump(o,pklfile)
        pklfile.close()



def dump_features(name, extractor,sub_dir):
    print('Extracting %s' % name)
    fnames = ['features_' + name, 'labels_' + name]
    test_features, test_labels = extractor(AUDIO_DIR, sub_dir)
    data = [( test_features), (test_labels)]
    dump_data(SAVE_DIR, data, fnames)

#dump_features('mean_'+str(1), extract_features_means,['test1'])

for i in TEST:
    print('On ',i)
    dump_features('mean_'+i, extract_features_means,[i])
    #dump_features('specs_'+i, extract_features_means,[i])
    #dump_features('specsNorm_'+i, extract_features_spectrograms_norm,[i])
    
















