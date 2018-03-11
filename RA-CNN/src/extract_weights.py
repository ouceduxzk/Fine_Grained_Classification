import sys
import os
import errno
import caffe
import layer_pb2
from easydict import EasyDict as edict 
import numpy as np 

def network(label, prototxt = None, modelpath=None):
    caffe.set_mode_cpu()
    import pdb; pdb.set_trace()
    base_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    models_dir = os.path.join(base_dir, 'models')
    arch_filepath = os.path.join(models_dir, '%s.prototxt' % label)
    weights_filepath = os.path.join(models_dir, '%s.caffemodel' % label)
    net = caffe.Net(arch_filepath if prototxt is None else prototxt, weights_filepath if modelpath is None else modelpath, caffe.TEST)
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
    layer = layer_pb2.Layer()
    layer.name = layer_name
    layer.shape.d1, layer.shape.d2, layer.shape.d3, layer.shape.d4 = \
        shape(weights)
    for value in weights.flatten():
        layer.weights.append(float(value))
    for value in bias.flatten():
        layer.bias.append(float(value))
    f.write(layer.SerializeToString())


def extract(label):
    net = network(label)
    base_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    layer_dir = os.path.join(base_dir, 'layer', label)
    mkdir_p(layer_dir)
    for layer_name in net.blobs:
        # import pdb; pdb.set_trace()
        if layer_name in net.params:
            weights, bias = net.params[layer_name]
            with open(os.path.join(layer_dir, layer_name), 'wb') as f:
                save_layer(f, layer_name, weights.data, bias.data)

def surgery(label, prototxt1, prototxt2):
    net_from = network(label, prototxt1)
    net_to   = network(label, prototxt2)
    base_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    layer_dir = os.path.join(base_dir, 'layer', label)
    for layer_name in net_from.blobs:
        if 'conv' not in layer_name : continue 
	print(layer_name)
        net_to.params[layer_name + '_A'][0].data[...] = net_from.params[layer_name][0].data[...]
        net_to.params[layer_name + '_A'][1].data[...] = net_from.params[layer_name][1].data[...]
        net_to.params[layer_name + '_A_A'][0].data[...] = net_from.params[layer_name][0].data[...]
        net_to.params[layer_name + '_A_A'][1].data[...] = net_from.params[layer_name][1].data[...]

    net_to.save('racnn.caffemodel')

def main(argv):
    #extract('VGG_ILSVRC_19_layers')
    #surgery('VGG_ILSVRC_19_layers', 'models/VGG_ILSVRC_19_layers.prototxt', 'models/surgery.prototxt')
    net = network('s', './proto/deploy_fixcls.prototxt', 'racnn.caffemodel')
if __name__ == '__main__':
    main(sys.argv)
