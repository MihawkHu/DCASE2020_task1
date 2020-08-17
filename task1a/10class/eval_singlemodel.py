import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import h5py
import scipy.io
import pandas as pd

import librosa
import soundfile as sound
import keras
import tensorflow

from sklearn.metrics import confusion_matrix

from utils import *
from funcs import *

from sklearn.metrics import log_loss

import os
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



#Task 1a dev validation set
val_csv = '/dockerdata/zhuhongning/dcase2020/evaluation_setup/fold1_evaluate.csv'
feat_path = '/dockerdata/rainiejjli/ft_local/dcase_task1_team-master/baseline/features/logmel128_scaled/'

# put the trained model path here
model_path = 'pretrained_models/10class-resnet-model-0.7458.hdf5'
# model_path = 'pretrained_models/10class-fcnn-model-0.7694.hdf5'
# model_path = 'pretrained_models/10class-fsfcnn-model-0.7620.hdf5'

dev_test_df = pd.read_csv(val_csv,sep='\t', encoding='ASCII')
wav_paths = dev_test_df['filename'].tolist()
ClassNames = np.unique(dev_test_df['scene_label'])

for idx, elem in enumerate(wav_paths):
    wav_paths[idx] = wav_paths[idx].split('/')[-1].split('.')[0]
    wav_paths[idx] = wav_paths[idx].split('-')[-1]

device_idxs = wav_paths
device_list = np.unique(device_idxs) 
print(device_list)

num_freq_bin = 128
num_classes = 10


# get results for each device
data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
data_deltas_val = deltas(data_val)
data_deltas_deltas_val = deltas(data_deltas_val)
data_val = np.concatenate((data_val[:,:,4:-4,:],data_deltas_val[:,:,2:-2,:],data_deltas_deltas_val),axis=-1)
y_val_onehot = keras.utils.to_categorical(y_val, num_classes)


best_model = keras.models.load_model(model_path)

preds = best_model.predict(data_val)
y_pred_val = np.argmax(preds,axis=1)

over_loss = log_loss(y_val_onehot, preds)
overall_acc = np.sum(y_pred_val==y_val) / data_val.shape[0]

print(y_val_onehot.shape, preds.shape)
np.set_printoptions(precision=3)

print("\n\nVal acc: ", "{0:.3f}".format(overall_acc))
print("Val log loss:", "{0:.3f}".format(over_loss))

device_acc = []
device_loss = []
for device_id in device_list:
    cur_preds = np.array([preds[i] for i in range(len(device_idxs)) if device_idxs[i] == device_id])
    cur_y_pred_val = np.argmax(cur_preds,axis=1)
    cur_y_val_onehot = np.array([y_val_onehot[i] for i in range(len(device_idxs)) if device_idxs[i] == device_id])
    cur_y_val = [y_val[i] for i in range(len(device_idxs)) if device_idxs[i] == device_id]
    cur_loss = log_loss(cur_y_val_onehot, cur_preds)
    cur_acc = np.sum(cur_y_pred_val==cur_y_val) / len(cur_preds)
    
    device_acc.append(cur_acc)
    device_loss.append(cur_loss)
    
print("\n\nDevices list: ", device_list)
print("Per-device val acc : ", np.array(device_acc))
print("Device A acc: ", "{0:.3f}".format(device_acc[0]))
print("Device B & C acc: ", "{0:.3f}".format((device_acc[1] + device_acc[2]) / 2))
print("Device s1 & s2 & s3 acc: ", "{0:.3f}".format((device_acc[3] + device_acc[4] + device_acc[5]) / 3))
print("Device s4 & s5 & s6 acc: ", "{0:.3f}".format((device_acc[6] + device_acc[7] + device_acc[8]) / 3))


# get confusion matrix
conf_matrix = confusion_matrix(y_val,y_pred_val)
print("\n\nConfusion matrix:")
print(conf_matrix)
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
mean_recall = np.mean(recall_by_class)

print("Class names:", ClassNames)
print("Per-class val acc: ",recall_by_class, "\n\n")





