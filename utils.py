import numpy as np
import _pickle as cPickle
import gzip
import os

MNIST_PATH = "data/mnist.pkl.gz"
CIFAR10_PATH = "data/cifar-10"

def one_hot_encode(y,num_class):
    m = y.shape[0]
    onehot = np.zeros((m,num_class),dtype="int32")
    for i in range(m):
        idx = y[i]
        onehot[i][idx] = 1
    return onehot

def accuracy(y_true,y_pred):
    return np.mean(y_pred == y_true) # both are not one hot encoded

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1,keepdims=True))
    return exp_x / np.sum(exp_x, axis=1,keepdims=True)

def load_mnist(num_training=1000,num_test=100,cnn=True,one_hot=False):
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
    X_train,y_train = X_train[range(num_training)],y_train[range(num_training)]
    X_test,y_test = X_test[range(num_test)],y_test[range(num_test)]
    return (X_train, y_train),(X_test, y_test)

def load_cifar10(num_training=1000,num_test=1000):
    Xs, ys = [], []
    for batch in range(1,6):
        f = open(os.path.join(CIFAR10_PATH,"data_batch_{0}".format(batch)),'rb')
        data = cPickle.load(f,encoding='iso-8859-1')
        f.close()
        X = data["data"].reshape(10000,3,32,32).astype("float64")
        y = np.array(data["labels"])
        Xs.append(X)
        ys.append(y)
    f = open(os.path.join(CIFAR10_PATH,"test_batch"),'rb')
    data = cPickle.load(f,encoding='iso-8859-1')
    f.close()
    X_train,y_train = np.concatenate(Xs),np.concatenate(ys)
    X_test = data["data"].reshape(10000,3,32,32).astype("float")
    y_test = np.array(data["labels"])
    X_train,y_train = X_train[range(num_training)],y_train[range(num_training)]
    X_test,y_test = X_test[range(num_test)],y_test[range(num_test)]
    mean = np.mean(X_train,axis=0)
    std = np.std(X_train)
    X_train = (X_train - mean) / std # Normalise data to [0, 1] range
    return (X_train, y_train),(X_test, y_test)

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def numerical_gradient_array(f, x, df, h=1e-5):
  """
  Evaluate a numeric gradient for a function that accepts a numpy
  array and returns a numpy array.
  """
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h
    pos = f(x).copy()
    x[ix] = oldval - h
    neg = f(x).copy()
    x[ix] = oldval

    grad[ix] = np.sum((pos - neg) * df) / (2 * h)

    it.iternext()
  return grad

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point

  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix])
    it.iternext() # step to next dimension

  return grad