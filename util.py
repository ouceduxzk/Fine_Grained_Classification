import numpy as np
from scipy import ndimage
import importlib

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

def load_model(fname):
    model = importlib.import_module('model_definitions.{}'.format(fname))
    return model


def get_mean_pixel(data):
    mean_pixel = np.mean(data.reshape(data.shape[0], data.shape[1] * data.shape[2]), 1)
    return mean_pixel
