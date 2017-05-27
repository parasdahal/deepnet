import numpy as np
import _pickle as cPickle
import gzip

MNIST_PATH = "data/mnist.pkl.gz"

def load_mnist(cnn=False,one_hot=False):
    f = gzip.open(MNIST_PATH,'rb')
    training_data, validation_data, test_data = cPickle.load(f,encoding='iso-8859-1')
    f.close()
    X_train, y_train = training_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data
    if cnn:
        shape = (-1,1,28,28)
        X_train = X_train.reshape(shape)
        X_validation = X_validation.reshape(shape)
        X_test = X_test.reshape(shape)
    if one_hot:
        y_train = one_hot_encode(y_train,10)
        y_validation = one_hot_encode(y_validation,10)
        y_test = one_hot_encode(y_test,10)
    return (X_train, y_train),(X_validation, y_validation),(X_test, y_test)

def one_hot_encode(y,num_class):
    m = y.shape[0]
    onehot = np.zeros((m,num_class),dtype="int32")
    for i in range(m):
        idx = y[i]
        onehot[i][idx] = 1
    return onehot

def accuracy(y_true,y_pred):
    # note that both y_true and y_pred are not one hot encoded
    return np.mean(y_pred == y_true)

def softmax(x):
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
