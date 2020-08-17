import os
import numpy as np
import scipy.io
#import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool
import librosa

overwrite = True

#file_path = '/nethome/hhu96/asc/2020_subtask_a/data_2020/'
#csv_file = '/nethome/hhu96/asc/2020_subtask_a/data_2020/evaluation_setup/fold1_all.csv'
#file_path = '/dockerdata/zhuhongning/data/dcase_audio/'
#file_path = '/apdcephfs/private_yannanwang/ft_local/mardy_rir/'
#file_path = '/apdcephfs/private_yannanwang/ft_local/train_rir_drc/'
file_path = '/dockerdata/rainiejjli/ft_local/dcase_task1_team-master/reverb_drc/2003_rir/'
#csv_file = '/dockerdata/zhuhongning/data/dcase_audio/'
output_path = 'features/logmel128_reverb_2003_scaled/'
feature_type = 'logmel'

def find_audio_files(path):
    """find audio files in given path"""
    files = []
    if not os.path.exists(path):
        print(path)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for root, dir_names, file_names in os.walk(path):
        for filename in file_names:
            if filename.endswith('.wav'):
                files.append(os.path.join(root, filename))
    return files


sr = 44100
duration = 10
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft/2)
num_time_bin = int(np.ceil(duration*sr/hop_length))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

#data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
#wavpath = data_df['filename'].tolist()
wavpath = find_audio_files(file_path)
print(len(wavpath))

for i in range(len(wavpath)):
    print(wavpath[i])
    #stereo, fs = sound.read(file_path + wavpath[i], stop=duration*sr)
    stereo, fs = librosa.load(wavpath[i], sr=sr)
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    #for ch in range(num_channel):
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

    logmel_data = np.log(logmel_data+1e-8)
    

    feat_data = logmel_data
    #print(feat_data.shape)
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    feature_data = {'feat_data': feat_data,}

    #cur_file_name = output_path + wavpath[i][5:-3] + feature_type
    cur_file_name = output_path +wavpath[i].split("/")[-1].split(".")[0]+".logmel"
    print(cur_file_name)
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        

