import numpy as np
import sys
import cPickle
from util import * 
from model_definitions.googlenet import * 

def load_train_data():
    train = np.load('car_train.npy') #np.memmap('data/X_train.npy', mode = 'r', shape = (7475,30, 192,192))
    y_train = np.load('y_train.npy')
    train = train.astype(np.float32)
    y_train = np.array(y_train, dtype = np.int32)
    print('train shape {}'.format(train.shape))
    return train/255.0, y_train

def load_test_data():
    test = np.load('car_test.npy')
    test.astype(np.float32)
    test = test/255.0
    return test

def split_data(X, y, split_ratio=0.2):
    split = int(X.shape[0] * split_ratio)
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]
    return X_train, y_train, X_test, y_test

def train_lasagne():
    model = build_model()
    #set_weights(net, 'models/res_march29.pkl')
    print('Loading training data...')
    X_train,  y_train = load_train_data()
    model.fit(X_train, y_train )

def test():
    model = build_model()
    net = model.model
    net.initialize()
    set_weights(net, 'models/res_april6.pkl')
    #net.predict_proba(X_test)
    X_test = load_test()
    y_score = net.predict_proba(X_test)
    with open('result.pkl', 'w') as fp :
 	cPickle.dump(y_score, fp)

train_lasagne()
#test(sys.argv[1])
