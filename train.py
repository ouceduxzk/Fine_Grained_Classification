from __future__ import print_function
import sys
import numpy as np
from vgg import *
import os
from util import *
from keras.utils import np_utils, generic_utils
from sklearn.metrics import accuracy_score
seed = 42

def load_train_data():
    train = np.load('data/train.npy')#np.memmap('data/X_train.npy', mode = 'r', shape = (7475,30, 192,192))
    test = np.load("data/test.npy")
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    train = train.astype(np.float32)
    test = test.astype(np.float32)
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2) 
    return train[:,np.newaxis, :,:], test[:,np.newaxis, :,:], y_train,y_test


def split_data(X, y, split_ratio=0.2):
    split = int(X.shape[0] * split_ratio)
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]
    return X_train, y_train, X_test, y_test


def train():
    model = build_vgg()
    X_train, X_test, y_train, y_test  = load_train_data()
    #model.fit(X_train, y_train , batch_size=32, nb_epoch=40, validation_data=(X_test , y_test))
    nb_iter = 100
    epochs_per_iter = 1
    batch_size = 128
    min_val_loss = sys.float_info.max
    X_train = preprocess(X_train)
    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)
        print('Fitting systole model...')
        X_train_aug =  rotation_augmentation(X_train, 15)
	X_train_aug =  shift_augmentation(X_train, 0.1,0.1)
        hist = model.fit(X_train_aug, y_train, shuffle=True, nb_epoch=epochs_per_iter, 
                                         batch_size=batch_size, validation_split=0.2, show_accuracy=True, verbose=2)
        train_loss = hist.history['loss'][-1]
        val_loss = hist.history['val_loss'][-1]
        print('Saving weights...')
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model.save_weights('weights/weight_best.hdf5', overwrite=True)
	if i %10 == 0:
           pred = model.predict(X_test)
           acc = accuracy_score(y_test, np.where(pred  > 0.5, 1 , 0))
           print('test acc {}'.format(acc))

def train_mean_pixel():
    train = np.load('data/train.npy')#np.memmap('data/X_train.npy', mode = 'r', shape = (7475,30, 192,192))
    test = np.load('data/test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    train = train.astype(np.float32)
    train_feature = get_mean_pixel(train)
    test_feature = get_mean_pixel(test)
    print(train_feature)
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import Ridge
    model = Ridge()
    model.fit(train_feature.reshape(50000,1), y_train)
    y_pred = model.predict(test_feature.reshape(50000,1))
    print(accuracy_score( y_test, np.where(y_pred > 0.5, 1, 0)))
