import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool


overwrite = True

file_path = '/nethome/hhu96/asc/2020_subtask_a/data_2020/'
csv_file = '/nethome/hhu96/asc/2020_subtask_a/data_2020/evaluation_setup/fold1_all.csv'
output_path = 'features/logmel128_scaled'
feature_type = 'logmel'

sr = 44100
SampleDuration = 10
NumFreqBins = 128
NumFFTPoints = 2048
HopLength = int(NumFFTPoints/2)
NumTimeBins = int(np.ceil(SampleDuration*sr/HopLength))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()


for i in range(len(wavpath)):
    stereo, fs = sound.read(file_path + wavpath[i], stop=SampleDuration*sr)
    logmel_data = np.zeros((NumFreqBins, NumTimeBins, num_channel), 'float32')
    #for ch in range(num_channel):
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=NumFFTPoints, hop_length=HopLength, n_mels=NumFreqBins, fmin=0.0, fmax=sr/2, htk=True, norm=None)

    logmel_data = np.log(logmel_data+1e-8)
    

    feat_data = logmel_data
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    feature_data = {'feat_data': feat_data,}

    cur_file_name = output_path + wavpath[i][5:-3] + feature_type
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        

