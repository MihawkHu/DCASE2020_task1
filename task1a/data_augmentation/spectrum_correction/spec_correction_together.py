import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import pandas as pd
import librosa
import soundfile as sound
import random
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pickle


file_path = 'data_2020/'
csv_file = file_path + 'evaluation_setup/fold1_train.csv'
val_csv_file = file_path + 'evaluation_setup/fold1_evaluate.csv'
device_a_csv = file_path + 'evaluation_setup/fold1_train_a.csv'

output_path = 'features/logmel128_scaled_speccorr'
feature_type = 'logmel'

if not os.path.exists(output_path):
    os.makedirs(output_path)


sr = 44100
num_audio_channels = 1
num_channel = 1
duration = 10

num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft / 2)
num_time_bin = int(np.ceil(duration * sr / hop_length))

dev_train_df = pd.read_csv(csv_file,sep='\t', encoding='ASCII')
dev_val_df = pd.read_csv(val_csv_file,sep='\t', encoding='ASCII')
wavpaths_train = dev_train_df['filename'].tolist()
wavpaths_val = dev_val_df['filename'].tolist()
y_train_labels =  dev_train_df['scene_label'].astype('category').cat.codes.values
y_val_labels =  dev_val_df['scene_label'].astype('category').cat.codes.values


trainf = [x[6:-4] for x in wavpaths_train]
train_subset = []

for idx in ['-a', '-b', '-c']:
    train_subset.append([x[:-1] for x in trainf if (x.endswith(idx))])

for idx in ['-s1', '-s2', '-s3']:
    train_subset.append([x[:-2] for x in trainf if (x.endswith(idx))])


train_sets=[]
for idx in range(len(train_subset)):
    train_sets.append(set(train_subset[idx]))

paired_wavs = []
# paired waves in subsets a, b, and c, s1, s2, and s3
for j in range(1, len(train_sets)):
    # paired waves in subsets b, c, and [s1, s2, s3] -- s4, s5, and s6 are not in the training partition of the data
    paired_wavs.append(train_sets[0] &  train_sets[j])

num_paired_wav = [ len(x) for x in paired_wavs]
min_paired_wav = 150

waves30 = []
wav_idxs = random.sample(range(min(num_paired_wav)), min_paired_wav)
for wavs in paired_wavs:
    temp = [list(wavs)[i] for i in wav_idxs]
    waves30.append(temp)


nbins_stft = int(np.ceil(num_fft/2.0)+1)
STFT_all = np.zeros((len(waves30)*min_paired_wav,nbins_stft,num_time_bin),'float32')
#STFT_all = np.zeros((min_paired_wav,nbins_stft,num_time_bin),'float32')
#STFT_sets = []
for group, x in zip(waves30, ['b', 'c', 's1','s2','s3']):
    i = 0
    for sc in group:
        wav_a = 'audio/' + sc + 'a.wav'
        wav_x = 'audio/' + sc + x + '.wav'
        stereo_a, fs = sound.read(file_path + wav_a, stop=duration * sr)
        stereo_x, fs = sound.read(file_path + wav_x, stop=duration * sr)
        # compute STFT of the paired signals
        STFT_a = librosa.stft(stereo_a, n_fft=num_fft, hop_length=hop_length)
        STFT_x = librosa.stft(stereo_x, n_fft=num_fft, hop_length=hop_length)
        # compute average value per each bin
        STFT_ref = np.abs(STFT_x)
        STFT_corr_coeff = STFT_ref/np.abs(STFT_a)
        # stack averaged values
        STFT_all[i,:,:] = STFT_corr_coeff
        i=i+1
    #STFT_sets.append(STFT_all)



STFT_hstak = np.hstack(STFT_all)
STFT_corr_coeff = np.expand_dims(np.mean(STFT_hstak,axis=1),-1)

data_df = pd.read_csv(device_a_csv, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()


#device_list = ['b', 'c', 's1','s2','s3']
for d in range(1):
    for i in range(len(wavpath)):
        stereo, fs = sound.read(file_path + wavpath[i], stop=duration*sr)
        STFT = librosa.stft(stereo, n_fft=num_fft, hop_length=hop_length)
        STFT_corr = np.abs(STFT)*STFT_corr_coeff
        
        logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
        logmel_data[:,:,0] = librosa.feature.melspectrogram(S=np.abs(STFT_corr)**2, sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

        logmel_data = np.log(logmel_data+1e-8)

        feat_data = logmel_data

        feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))

        feature_data = {'feat_data': feat_data,}
        cur_file_name = output_path + wavpath[i][5:-4] + '-all.'  + feature_type
        pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
    



