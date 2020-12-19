import keras
from keras import backend as K
import numpy as np
import threading
import pandas
import sys
sys.path.append("..")
from funcs import *
from utils import *

class LR_WarmRestart(keras.callbacks.Callback):
    
    def __init__(self,nbatch,initial_lr,min_lr,epochs_restart,Tmult):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs_restart = epochs_restart
        self.nbatch = nbatch
        self.currentEP = 0
        self.startEP = 0
        self.Tmult = Tmult
        
    def on_epoch_begin(self, epoch, logs={}):
        if epoch+1<self.epochs_restart[0]:
            self.currentEP = epoch
        else:
            self.currentEP = epoch+1
            
        if np.isin(self.currentEP,self.epochs_restart):
            self.startEP=self.currentEP
            self.Tmult=2*self.Tmult
        
    def on_epoch_end(self, epochs, logs={}):
        lr = K.get_value(self.model.optimizer.lr)
        print ('\nLearningRate:{:.6f}'.format(lr))
    
    def on_batch_begin(self, batch, logs={}):
        pts = self.currentEP + batch/self.nbatch - self.startEP
        decay = 1+np.cos(pts/self.Tmult*np.pi)
        lr = self.min_lr+0.5*(self.initial_lr-self.min_lr)*decay
        K.set_value(self.model.optimizer.lr,lr)

        
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
        
        
def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class Generator_withdelta_splitted():
    def __init__(self, feat_path, train_csv, feat_dim, batch_size=32, alpha=0.2, shuffle=True, crop_length=400, splitted_num=4): 
        self.feat_path = feat_path
        self.train_csv = train_csv
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(open(train_csv, 'r').readlines()) - 1
        self.lock = threading.Lock()
        self.NewLength = crop_length
        self.splitted_num = splitted_num
        
    def __iter__(self):
        return self
    
    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()

                # split data and then load it to memory one by one
                item_num = self.sample_num // self.splitted_num - (self.sample_num // self.splitted_num) % self.batch_size
                for k in range(self.splitted_num):
                    cur_item_num = item_num
                    s = k * item_num
                    e = (k+1) * item_num 
                    if k == self.splitted_num - 1:
                        cur_item_num = self.sample_num - (self.splitted_num - 1) * item_num
                        e = self.sample_num

                    lines = indexes[s:e]
                    X_train, y_train = load_data_2020_splitted(self.feat_path, self.train_csv, self.feat_dim, lines, 'logmel')
                    y_train = keras.utils.to_categorical(y_train, 3)
                    X_deltas_train = deltas(X_train)
                    X_deltas_deltas_train = deltas(X_deltas_train)
                    X_train = np.concatenate((X_train[:,:,4:-4,:],X_deltas_train[:,:,2:-2,:],X_deltas_deltas_train),axis=-1)

                    
                    itr_num = int(cur_item_num // (self.batch_size * 2))
                    for i in range(itr_num):
                        batch_ids = np.arange(cur_item_num)[i*self.batch_size * 2:(i + 1) * self.batch_size * 2]
                        X, y = self.__data_generation(batch_ids, X_train, y_train)
                        
                        yield X, y


    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids, X_train, y_train):
        _, h, w, c = X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = X_train[batch_ids[:self.batch_size]]
        X2 = X_train[batch_ids[self.batch_size:]]
        
        # random cropping
        for j in range(X1.shape[0]):
            StartLoc1 = np.random.randint(0,X1.shape[2]-self.NewLength)
            StartLoc2 = np.random.randint(0,X2.shape[2]-self.NewLength)

            X1[j,:,0:self.NewLength,:] = X1[j,:,StartLoc1:StartLoc1+self.NewLength,:]
            X2[j,:,0:self.NewLength,:] = X2[j,:,StartLoc2:StartLoc2+self.NewLength,:]
            
        X1 = X1[:,:,0:self.NewLength,:]
        X2 = X2[:,:,0:self.NewLength,:]
        
        # mixup
        X = X1 * X_l + X2 * (1.0 - X_l)

        if isinstance(y_train, list):
            y = []

            for y_train_ in y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1.0 - y_l))
        else:
            y1 = y_train[batch_ids[:self.batch_size]]
            y2 = y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1.0 - y_l)

        return X, y

