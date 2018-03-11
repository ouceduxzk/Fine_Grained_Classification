import sys
import os
import errno
import caffe
import proto.layer_pb2
from network import network
from easydict import EasyDict as edict 
import numpy as np 

def network(label):
    caffe.set_mode_cpu()
    base_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    models_dir = os.path.join(base_dir, 'models')
    arch_filepath = os.path.join(models_dir, '%s.prototxt' % label)
    weights_filepath = os.path.join(models_dir, '%s.caffemodel' % label)
    net = caffe.Net(arch_filepath, weights_filepath, caffe.TEST)
    return net 

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def shape(arr):
    shape = [1, 1, 1, 1]
    shape[-len(arr.shape):] = arr.shape
    return shape


def save_layer(f, layer_name, weights, bias):
    print('Saving %s' % layer_name)
    layer = proto.layer_pb2.Layer()
    layer.name = layer_name
    layer.shape.d1, layer.shape.d2, layer.shape.d3, layer.shape.d4 = \
        shape(weights)
    for value in weights.flatten():
        layer.weights.append(float(value))
    for value in bias.flatten():
        layer.bias.append(float(value))
    f.write(layer.SerializeToString())


def extract():
    net = network(label)
    base_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    layer_dir = os.path.join(base_dir, 'layer', label)
    mkdir_p(layer_dir)
    # image_filepath = os.path.join(base_dir, 'face.png')
    # net = net.forward(image_filepath)

    for layer_name in net.blobs:
        # import pdb; pdb.set_trace()
        if layer_name in net.params:
            weights, bias = net.params[layer_name]
            with open(os.path.join(layer_dir, layer_name), 'wb') as f:
                save_layer(f, layer_name, weights.data, bias.data)

def main(argv):
    extract('VGG_ILSVRC_19_layers')

if __name__ == '__main__':
    main(sys.argv)
