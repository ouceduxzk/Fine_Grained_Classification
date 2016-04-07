# BLVC Googlenet, model from the paper:
# "Going Deeper with Convolutions"
# Original source:
# https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
# License: unrestricted use

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl
import sys
sys.path.insert(0, '/home/zaikun/scratch/kaggle/nn/statefarm/scripts/')
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.layers import set_all_param_values
from utils.nolearn_net import NeuralNet
from nolearn.lasagne.handlers import SaveWeights
import lasagne 
import theano
# from nolearn_utils.iterators import (
#     ShuffleBatchIteratorMixin,
#     BufferedBatchIteratorMixin,
#     RandomCropBatchIteratorMixin,
#     RandomFlipBatchIteratorMixin,
#     AffineTransformBatchIteratorMixin,
#     AdjustGammaBatchIteratorMixin,
#     make_iterator
# )

from nolearn_utils.hooks import (
    SaveTrainingHistory,
    PlotTrainingHistory,
    EarlyStopping,
    StepDecay
)
#from nolearn.lasagne import TrainSplit
from utils import TrainSplit
from utils.layer_macros import conv2dbn
from utils.layer_macros import residual_block3_localbn as residual_block
from utils.customer import * 

def float32(k):
    return np.cast['float32'](k)

model_fname = './models/gnet_ap4.pkl'
model_history_fname = './models/gnet_history.pkl'
model_graph_fname = './models/gnet_plot.png'

image_size = 224
batch_size = 32

save_weights = SaveWeights(model_fname, only_best=True, pickle=False)
save_training_history = SaveTrainingHistory(model_history_fname)
plot_training_history = PlotTrainingHistory(model_graph_fname)
early_stopping = EarlyStopping(patience=20)


root = '/home/zaikun/scratch/kaggle/nn/car/'
def build_inception_module(name, input_layer, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(
        net['pool'], nfilters[0], 1, flip_filters=False)

    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

    net['3x3_reduce'] = ConvLayer(
        input_layer, nfilters[2], 1, flip_filters=False)
    net['3x3'] = ConvLayer(
        net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

    net['5x5_reduce'] = ConvLayer(
        input_layer, nfilters[4], 1, flip_filters=False)
    net['5x5'] = ConvLayer(
        net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
        ])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, image_size, image_size))
    net['conv1/7x7_s2'] = ConvLayer(
        net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = PoolLayer(
        net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(
        net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(
        net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(net['conv2/norm2'], pool_size=3, stride=2)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = PoolLayer(net['inception_3b/output'],
                                    pool_size=3, stride=2)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = PoolLayer(net['inception_4e/output'],
                                    pool_size=3, stride=2)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))

    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    import pickle
    model = pickle.load(open(root + 'models/blvc_googlenet.pkl'))
    set_all_param_values(net['pool5/7x7_s1'], model['param values'][:114])    
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                         num_units=196,
                                         nonlinearity=linear)
    net['prob'] = NonlinearityLayer(net['loss3/classifier'],
                                    nonlinearity=softmax)

    model = NeuralNet(
        layers=net['prob'],
        #use_label_encoder=False,
        #objective_l2=1e-4, #1e-3
        #update=lasagne.updates.adam,
        #update_learning_rate=1e-4,
        update=lasagne.updates.nesterov_momentum,
        update_momentum=0.9,
        update_learning_rate=theano.shared(float32(0.03)), # 1e-4
        train_split=TrainSplit(0.1, random_state=42, stratify=False),
        #batch_iterator_train=train_iterator,
        #batch_iterator_test=test_iterator,
        on_epoch_finished=[
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping,
            #StepDecay('update_learning_rate', start=1e-2, stop=1e-3)
        ],
        verbose=1,
        max_epochs=200,
        #custom_score = ('CRPS', CRPS)
    )

    return model
