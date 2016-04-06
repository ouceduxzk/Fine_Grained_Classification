import numpy as np
import importlib
import sys
import cPickle
from vgg import * 
from skimage.filters import sobel

def set_weights(net,model_file):
    '''
    Sets the parameters of the model using the weights stored in model_file
    Parameters
    ----------
    net: a Lasagne layer
    model_file: string
        path to the model that containes the weights
    Returns
    -------
    None
    '''
    with open(model_file) as f:
        print('Load pretrained weights from %s...' % model_file)
        model = cPickle.load(f)

    print('Set the weights...')
    net.load_params_from(model)

def load_train_data():
    train = np.load('car_train.npy') #np.memmap('data/X_train.npy', mode = 'r', shape = (7475,30, 192,192))
    y_train = np.load('y_train.npy')
    train = train[:, np.newaxis, :,:].astype(np.float32)
    y_train = np.array(y_train, dtype = np.int32)
    print('train shape {}'.format(train.shape))
    return train, y_train

def load_model(fname):
    model = importlib.import_module('model_definitions.{}'.format(fname))
    return model

def split_data(X, y, split_ratio=0.2):
    split = int(X.shape[0] * split_ratio)
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]
    return X_train, y_train, X_test, y_test


def train_lasagne(model_name):
    model = load_model(model_name)
    net = model.net
    net.initialize()
    #set_weights(net, 'models/res_march29.pkl')
    print('Loading training data...')
    X_train,  y_train = load_train_data()
    net.fit(X_train, y_train )

def test(model_name):
    model = load_model(model_name)
    net = model.net
    net.initialize()
    set_weights(net, 'models/res_march29.pkl')
    #net.predict_proba(X_test)
    X_train, X_test, y_train, y_test  = load_train_data()
    y_score = net.predict_proba(X_test)
    y_test = np_utils.to_categorical(y_test, 2)
    print(y_score.shape)
    print(y_test.shape)
    with open('result.pkl', 'w') as fp :
 	cPickle.dump(y_score, fp)
        cPickle.dump(y_test, fp)


train_lasagne(sys.argv[1])
#test(sys.argv[1])
