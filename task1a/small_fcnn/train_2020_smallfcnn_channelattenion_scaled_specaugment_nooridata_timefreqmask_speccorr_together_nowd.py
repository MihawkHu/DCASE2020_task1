# %%
# select a GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
#imports 
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import librosa
import soundfile as sound

import keras
import tensorflow
from keras.optimizers import SGD

from utils import *
from funcs import *

from DCASE2020_FCNNCHTT import model_fcnn
from DCASE_training_functions import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ThisPath = '/nethome/hhu96/asc/2020_subtask_a/data_2020/'
train_csv = ThisPath + 'evaluation_setup/fold1_train.csv'
val_csv = ThisPath + 'evaluation_setup/fold1_evaluate.csv'
feat_path = '/nethome/hhu96/asc/2020_subtask_a/features/logmel128_scaled_d_dd/'
aug_path = '/nethome/hhu96/asc/2020_subtask_a/features/logmel128_scaled_d_dd_speccorr_together/'
aug_csv =  ThisPath + 'evaluation_setup/fold1_train_speccorr_together.csv'
experiments = 'exp_2020_smallfcnn_channelattention_scaled_specaugment_nooridata_timefreqmask_speccorr_together_nowd'

if not os.path.exists(experiments):
    os.makedirs(experiments)


train_aug_csv = generate_train_aug_csv(train_csv, aug_csv, feat_path, aug_path, experiments)

num_audio_channels = 1
NumFreqBins = 128
NumClasses = 10
max_lr = 0.1
batch_size = 32
num_epochs = 254
mixup_alpha = 0.4
crop_length = 400
sample_num = len(open(train_aug_csv, 'r').readlines()) - 1



LM_val, y_val = load_data_2020(feat_path, val_csv, NumFreqBins, 'logmel')
y_val = keras.utils.to_categorical(y_val, NumClasses)

model = model_fcnn(NumClasses, input_shape=[NumFreqBins, None, 3*num_audio_channels], num_filters=[48, 96, 192], wd=0)

model.compile(loss='categorical_crossentropy',
              optimizer =SGD(lr=max_lr,decay=0, momentum=0.9, nesterov=False),
              metrics=['accuracy'])

#model.summary()

#set learning rate schedule
lr_scheduler = LR_WarmRestart(nbatch=np.ceil(sample_num/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0]) 
save_path = experiments + "/saved-model-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks = [lr_scheduler, checkpoint]

#create data generator
TrainDataGen = MixupGenerator_timefreqmask_withaug_splitted(train_aug_csv, NumFreqBins, 
                              batch_size=batch_size,
                              alpha=mixup_alpha,
                              crop_length=crop_length, splitted_num=4)()

#train the model
print("training start.")
history = model.fit_generator(TrainDataGen,
                              validation_data=(LM_val, y_val),
                              epochs=num_epochs, 
                              verbose=1, 
                              workers=4,
                              max_queue_size = 100,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(sample_num/batch_size)
                              ) 

