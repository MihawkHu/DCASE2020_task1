import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import keras
import tensorflow
from keras.optimizers import SGD

import sys
sys.path.append("..")
from utils import *
from funcs import *

from fcnn_att import model_fcnn
from DCASE_training_functions import *

from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Please put your csv file for train and validation here.
# If you dont generate the extra augmented data, please use 
# ../evaluation_setup/fold1_train.csv and delete the aug_csv part
train_csv = 'evaluation_setup/fold1_train_full.csv'
val_csv = 'evaluation_setup/fold1_evaluate.csv'
aug_csv = 'evaluation_setup/fold1_train_a_2003.csv'

feat_path = 'features/logmel128_scaled_full/'
aug_path = 'features/logmel128_reverb_scaled/'

experiments = 'exp_2020_fcnn_channelattention_scaled_specaugment_timefreqmask_speccorr_together_nowd_alldata/'

if not os.path.exists(experiments):
    os.makedirs(experiments)


train_aug_csv = generate_train_aug_csv(train_csv, aug_csv, feat_path, aug_path, experiments)

num_audio_channels = 1
num_freq_bin = 128
num_classes = 10
max_lr = 0.1
batch_size = 32
num_epochs = 254
mixup_alpha = 0.4
crop_length = 400
sample_num = len(open(train_aug_csv, 'r').readlines()) - 1


# compute delta and delta delta for validation data
data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
data_deltas_val = deltas(data_val)
data_deltas_deltas_val = deltas(data_deltas_val)
data_val = np.concatenate((data_val[:,:,4:-4,:],data_deltas_val[:,:,2:-2,:],data_deltas_deltas_val),axis=-1)
y_val = keras.utils.to_categorical(y_val, num_classes)

model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, 3*num_audio_channels], num_filters=[48, 96, 192], wd=0)

model.compile(loss='categorical_crossentropy',
              optimizer =SGD(lr=max_lr,decay=1e-6, momentum=0.9, nesterov=False),
              metrics=['accuracy'])

model.summary()

lr_scheduler = LR_WarmRestart(nbatch=np.ceil(sample_num/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0]) 
save_path = experiments + "/model-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks = [lr_scheduler, checkpoint]

# Due to the memory limitation, in the training stage we split the training data
train_data_generator = Generator_timefreqmask_withdelta_splitted(feat_path, train_aug_csv, num_freq_bin,
                              batch_size=batch_size,
                              alpha=mixup_alpha,
                              crop_length=crop_length, splitted_num=20)()

history = model.fit_generator(train_data_generator,
                              validation_data=(data_val, y_val),
                              epochs=num_epochs, 
                              verbose=1, 
                              workers=4,
                              max_queue_size = 100,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(sample_num/batch_size)
                              ) 

