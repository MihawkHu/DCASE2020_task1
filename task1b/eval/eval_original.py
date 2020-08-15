##### For testing the original keras model, which is saved as .hdf5 format.

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
from sklearn.metrics import log_loss
import sys
sys.path.append("..")
from utils import *
from funcs import *

from tensorflow import ConfigProto
from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

val_csv = 'data_2020/evaluation_setup/fold1_evaluate.csv'
feat_path = 'features/logmel128_scaled_d_dd/'
model_path = '../pretrained_models/smallfcnn-model-0.9618.hdf5'

num_freq_bin = 128
num_classes = 3

data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
y_val_onehot = keras.utils.to_categorical(y_val, num_classes)

print(data_val.shape)
print(y_val.shape)

best_model = keras.models.load_model(model_path)
preds = best_model.predict(data_val)

y_pred_val = np.argmax(preds,axis=1)

over_loss = log_loss(y_val_onehot, preds)
overall_acc = np.sum(y_pred_val==y_val) / data_val.shape[0]

print(y_val_onehot.shape, preds.shape)
np.set_printoptions(precision=3)

print("\n\nVal acc: ", "{0:.3f}".format(overall_acc))
print("Val log loss:", "{0:.3f}".format(over_loss))

conf_matrix = confusion_matrix(y_val,y_pred_val)
print("\n\nConfusion matrix:")
print(conf_matrix)
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
mean_recall = np.mean(recall_by_class)

dev_test_df = pd.read_csv(val_csv,sep='\t', encoding='ASCII')
ClassNames = np.unique(dev_test_df['scene_label'])

print("Class names:", ClassNames)
print("Per-class val acc: ",recall_by_class, "\n\n")





