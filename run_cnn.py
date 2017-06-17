import numpy as np
from deepnet.utils import load_mnist, load_cifar10
from deepnet.layers import *
from deepnet.solver import sgd, sgd_momentum, adam
from deepnet.nnet import CNN
import sys


def make_mnist_cnn(X_dim, num_class):
    conv = Conv(X_dim, n_filter=32, h_filter=3,
                w_filter=3, stride=1, padding=1)
    relu_conv = ReLU()
    maxpool = Maxpool(conv.out_dim, size=2, stride=1)
    flat = Flatten()
    fc = FullyConnected(np.prod(maxpool.out_dim), num_class)
    return [conv, relu_conv, maxpool, flat, fc]


def make_cifar10_cnn(X_dim, num_class):
    conv = Conv(X_dim, n_filter=16, h_filter=5,
                w_filter=5, stride=1, padding=2)
    relu = ReLU()
    maxpool = Maxpool(conv.out_dim, size=2, stride=2)
    conv2 = Conv(maxpool.out_dim, n_filter=20, h_filter=5,
                 w_filter=5, stride=1, padding=2)
    relu2 = ReLU()
    maxpool2 = Maxpool(conv2.out_dim, size=2, stride=2)
    flat = Flatten()
    fc = FullyConnected(np.prod(maxpool2.out_dim), num_class)
    return [conv, relu, maxpool, conv2, relu2, maxpool2, flat, fc]


if __name__ == "__main__":

    if sys.argv[1] == "mnist":

        training_set, test_set = load_mnist(
            'data/mnist.pkl.gz', num_training=1000, num_test=1000)
        X, y = training_set
        X_test, y_test = test_set
        mnist_dims = (1, 28, 28)
        cnn = CNN(make_mnist_cnn(mnist_dims, num_class=10))
        cnn = sgd_momentum(cnn, X, y, minibatch_size=35, epoch=20,
                           learning_rate=0.01, X_test=X_test, y_test=y_test)

    if sys.argv[1] == "cifar10":
        training_set, test_set = load_cifar10(
            'data/cifar-10', num_training=1000, num_test=100)
        X, y = training_set
        X_test, y_test = test_set
        cifar10_dims = (3, 32, 32)
        cnn = CNN(make_cifar10_cnn(cifar10_dims, num_class=10))
        cnn = sgd_momentum(cnn, X, y, minibatch_size=10, epoch=200,
                           learning_rate=0.01, X_test=X_test, y_test=y_test)
