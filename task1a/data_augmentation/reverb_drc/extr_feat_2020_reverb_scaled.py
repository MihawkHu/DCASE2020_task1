import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool


overwrite = True

file_path = 'mardy_drc/'
csv_file = 'evaluation_setup/fold1_train_a.csv'
output_path = 'features/logmel128_reverb_scaled'

feature_type = 'logmel'

sr = 44100
duration = 10
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft/2)
num_time_bin = int(np.ceil(duration*sr/hop_length))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()

# def cal_deltas(X_in):
#     X_out = (X_in[:,2:,:]-X_in[:,:-2,:])/10.0
#     X_out = X_out[:,1:-1,:]+(X_in[:,4:,:]-X_in[:,:-4,:])/5.0
#     return X_out


for i in range(len(wavpath)):
    stereo, fs = sound.read(file_path + wavpath[i], stop=duration*sr)
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

    logmel_data = np.log(logmel_data+1e-8)

    feat_data = logmel_data
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    
    # deltas = cal_deltas(feat_data)
    # deltas_deltas = cal_deltas(deltas)
    # feat_data = np.concatenate((feat_data[:,4:-4,:], deltas[:,2:-2,:], deltas_deltas), axis=2)

    feature_data = {'feat_data': feat_data,}

    # use MARDY as example, please use different name for different generated reverberation data
    cur_file_name = output_path + wavpath[i][5:-4] + '_mardy_drc.' + feature_type
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        

