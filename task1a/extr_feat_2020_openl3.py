import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool

import openl3


overwrite = True

file_path = '/nethome/hhu96/asc/2020_subtask_a/data_2020/'
csv_file = '/nethome/hhu96/asc/2020_subtask_a/data_2020/evaluation_setup/fold1_all.csv'
output_path = 'features/openl3_mel256_hop20_d512'
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

def cal_deltas(X_in):
    X_out = (X_in[:,2:,:]-X_in[:,:-2,:])/10.0
    X_out = X_out[:,1:-1,:]+(X_in[:,4:,:]-X_in[:,:-4,:])/5.0
    return X_out


for i in range(len(wavpath)):
    stereo, fs = sound.read(file_path + wavpath[i], stop=SampleDuration*sr)
    #logmel_data = np.zeros((NumFreqBins, NumTimeBins, num_channel), 'float32')
    #logmel_data[:,:, 0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=NumFFTPoints, hop_length=HopLength, n_mels=NumFreqBins, fmin=0.0, fmax=sr/2, htk=True, norm=None)

    emb, ts = openl3.get_audio_embedding(stereo, sr, content_type="env", input_repr="mel256", embedding_size=512, hop_size=0.02, verbose=0)


    #logmel_data = np.log(logmel_data+1e-8)
    
    #deltas = cal_deltas(logmel_data)
    #deltas_deltas = cal_deltas(deltas)

    #feat_data = np.concatenate((logmel_data[:,4:-4,:], deltas[:,2:-2,:], deltas_deltas), axis=2)
    feat_data = emb
    feature_data = {'feat_data': feat_data,}

    cur_file_name = output_path + wavpath[i][5:-3] + feature_type
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print(i)
        
        

