import numpy as np
import pickle
import random
import pandas as pd


def frequency_masking(mel_spectrogram, frequency_masking_para=13, frequency_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(frequency_mask_num):
        f = random.randrange(0, frequency_masking_para)
        f0 = random.randrange(0, fbank_size[0] - f)

        if (f0 == f0 + f):
            continue

        mel_spectrogram[f0:(f0+f),:] = 0
    return mel_spectrogram
   
   
def time_masking(mel_spectrogram, time_masking_para=40, time_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(time_mask_num):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, fbank_size[1] - t)

        if (t0 == t0 + t):
            continue

        mel_spectrogram[:, t0:(t0+t)] = 0
    return mel_spectrogram



def cmvn(data):
    shape = data.shape
    eps = 2**-30
    for i in range(shape[0]):
        utt = data[i].squeeze().T
        mean = np.mean(utt, axis=0)
        utt = utt - mean
        std = np.std(utt, axis=0)
        utt = utt / (std + eps)
        utt = utt.T
        data[i] = utt.reshape((utt.shape[0], utt.shape[1], 1))
    return data


def sample_csv(total_csv, sample_num, experiments):
    f = open(total_csv, 'r').readlines()
    title = f[0]
    f = f[1:]

    total_len = len(f)
    selected_idx = random.sample(range(total_len), sample_num)

    res_csv = experiments + '/res.csv'
    fw = open(res_csv, 'w')
    fw.write(title)
    for i in range(sample_num):
        fw.write(f[selected_idx[i]])

    fw.close()

    return res_csv

            
def balance_class_data(train_csv, experiments):
    f = open(train_csv, 'r').readlines()
    title = f[0]
    f = f[1:]

    lines = f.copy()
    for idx, elem in enumerate(lines):
        lines[idx] = lines[idx].split('\t')
        lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

    class_idxs = [lines[i][1] for i in range(len(f))]
    class_list = np.unique(class_idxs)
    
    class_pos = []
    num_per_class = []
    for class_id in class_list:
        cur_list = [i for i in range(len(class_idxs)) if class_idxs[i] == class_id]
        class_pos.append(cur_list)
        num_per_class.append(len(cur_list))

    max_num_per_class = max(num_per_class)
    
    class_sample_list = []
    for i in range(len(class_list)):
        cur_sample_list = random.choices(class_pos[i], k=max_num_per_class)
        class_sample_list.append(cur_sample_list)

    res_csv = experiments + '/' + train_csv.split('/')[-1][:-4] + '_balanceclass.csv'
    fw = open(res_csv, 'w')
    fw.write(title)
    for i in range(len(class_list)):
        cur_sample_list = class_sample_list[i]
        for j in range(len(cur_sample_list)):
            cur_idx = cur_sample_list[j]
            fw.write(f[cur_idx])
    fw.close()

    return res_csv
            
