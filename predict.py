import numpy as np
from keras.utils import np_utils, generic_utils
from vgg import *

def predict():
    test = np.load('data/test.npy')
    y_test = np.load('data/y_test.npy')
    y_test = np_utils.to_categorical(y_test, 2)
    model = build_vgg('weights/weight_best.hdf5')
    pred = model.predict(test[:,np.newaxis,:,:].astype(np.float32), batch_size=128, show_accuracy=True, verbose=2)
    #print pred, y_test

