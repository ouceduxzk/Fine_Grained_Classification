# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax


def build_model(): 
    inp = InputLayer((None, 3, 224, 224))
    net = ConvLayer( inp, 64, 3, pad=1, flip_filters=False)
    net= ConvLayer( net,  64, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net  = ConvLayer(
        net, 128, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 128, 3, pad=1, flip_filters=False)
    net =  PoolLayer(net, 2)
    net = ConvLayer(
        net, 256, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 256, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 256, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = ConvLayer(
        net, 512, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 512, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 512, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = ConvLayer(
        net, 512, 3, pad=1, flip_filters=False)
    net = ConvLayer(
        net, 512, 3, pad=1, flip_filters=False)
    net =  ConvLayer(
        net, 512, 3, pad=1, flip_filters=False)
    net = PoolLayer(net, 2)
    net = DenseLayer(net, num_units=1000)
    net= DropoutLayer(net, p=0.5)
    net = DenseLayer(net, num_units=1000)
    net = DropoutLayer(net, p=0.5)
    net = DenseLayer(
        net, num_units=196, nonlinearity = softmax)

    return [inp], net
