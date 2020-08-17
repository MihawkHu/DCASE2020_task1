import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool


overwrite = True

file_path = 'data_2020/'
csv_file = 'evaluation_setup/fold1_train.csv'
output_path = 'features/logmel128_scaled_full'

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


for i in range(len(wavpath)):
    stereo, fs = sound.read(file_path + wavpath[i], stop=duration*sr)
    time_factor = np.random.uniform(0.5, 2)
    length = len(stereo)
    y_stretch = librosa.effects.time_stretch(stereo, time_factor)
    if len(y_stretch) < length:
        y_stretch = np.concatenate((y_stretch, y_stretch))
        y_stretch = y_stretch[0:length]
    else:
        y_stretch = y_stretch[0:length]
    stereo = y_stretch
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

    logmel_data = np.log(logmel_data+1e-8)

    feat_data = logmel_data
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))

    feature_data = {'feat_data': feat_data,}

    cur_file_name = output_path + wavpath[i][5:-4] + '_time.' + feature_type
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        

