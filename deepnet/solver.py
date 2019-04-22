import numpy as np
from sklearn.utils import shuffle
from deepnet.utils import accuracy
import copy
from deepnet.loss import SoftmaxLoss


def get_minibatches(X, y, minibatch_size,shuffleTag=True):
    m = X.shape[0]
    minibatches = []
    if shuffleTag:
        X, y = shuffle(X, y)
    for i in range(0, m, minibatch_size):
        X_batch = X[i:i + minibatch_size, :, :, :]
        y_batch = y[i:i + minibatch_size, ]
        minibatches.append((X_batch, y_batch))
    return minibatches


def vanilla_update(params, grads, learning_rate=0.01):
    for param, grad in zip(params, reversed(grads)):
        for i in range(len(grad)):
            param[i] += - learning_rate * grad[i]


def momentum_update(velocity, params, grads, learning_rate=0.01, mu=0.9):
    for v, param, grad, in zip(velocity, params, reversed(grads)):
        for i in range(len(grad)):
            v[i] = mu * v[i] + learning_rate * grad[i]
            param[i] -= v[i]


def adagrad_update(cache, params, grads, learning_rate=0.01):
    for c, param, grad, in zip(cache, params, reversed(grads)):
        for i in range(len(grad)):
            cache[i] += grad[i]**2
            param[i] += - learning_rate * grad[i] / (np.sqrt(cache[i]) + 1e-8)


def rmsprop_update(cache, params, grads, learning_rate=0.01, decay_rate=0.9):
    for c, param, grad, in zip(cache, params, reversed(grads)):
        for i in range(len(grad)):
            cache[i] = decay_rate * cache[i] + (1 - decay_rate) * grad[i]**2
            param[i] += - learning_rate * grad[i] / (np.sqrt(cache[i]) + 1e-4)


def sgd(nnet, X_train, y_train, minibatch_size, epoch, learning_rate, verbose=True,
        X_test=None, y_test=None):
    minibatches = get_minibatches(X_train, y_train, minibatch_size)
    for i in range(epoch):
        loss = 0
        if verbose:
            print("Epoch {0}".format(i + 1))
        for X_mini, y_mini in minibatches:
            loss, grads = nnet.train_step(X_mini, y_mini)
            vanilla_update(nnet.params, grads, learning_rate=learning_rate)
        if verbose:
            train_acc = accuracy(y_train, nnet.predict(X_train))
            test_acc = accuracy(y_test, nnet.predict(X_test))
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".format(
                loss, train_acc, test_acc))
    return nnet

def sgd_rnn(nnet, X_train, y_train, minibatch_size, epoch, learning_rate, verbose=True):
    for i in range(epoch):
        loss = 0
        if verbose:
            print("Epoch {0}".format(i + 1))
        hidden_state = nnet.initial_state
        loss, grads, hidden_state = nnet.train_step(X_train, y_train, hidden_state)

        for k in grads.keys():
            nnet.model[k] -= learning_rate * grads[k]
        
        if verbose:
            print("Loss = {0}".format(loss))
    return nnet


def sgd_momentum(nnet, X_train, y_train, minibatch_size, epoch, learning_rate, mu=0.9,
                 verbose=True, X_test=None, y_test=None, nesterov=True):

    minibatches = get_minibatches(X_train, y_train, minibatch_size)

    for i in range(epoch):
        loss = 0
        velocity = []
        for param_layer in nnet.params:
            p = [np.zeros_like(param) for param in list(param_layer)]
            velocity.append(p)

        if verbose:
            print("Epoch {0}".format(i + 1))

        for X_mini, y_mini in minibatches:

            if nesterov:
                for param, ve in zip(nnet.params, velocity):
                    for i in range(len(param)):
                        param[i] += mu * ve[i]

            loss, grads = nnet.train_step(X_mini, y_mini)
            momentum_update(velocity, nnet.params, grads,
                            learning_rate=learning_rate, mu=mu)

        if verbose:
            m_train = X_train.shape[0]
            m_test = X_test.shape[0]
            y_train_pred = np.array([], dtype="int64")
            y_test_pred = np.array([], dtype="int64")
            for i in range(0, m_train, minibatch_size):
                X_tr = X_train[i:i + minibatch_size, :, :, :]
                y_tr = y_train[i:i + minibatch_size, ]
                y_train_pred = np.append(y_train_pred, nnet.predict(X_tr))
            for i in range(0, m_test, minibatch_size):
                X_te = X_test[i:i + minibatch_size, :, :, :]
                y_te = y_test[i:i + minibatch_size, ]
                y_test_pred = np.append(y_test_pred, nnet.predict(X_te))

            train_acc = accuracy(y_train, y_train_pred)
            test_acc = accuracy(y_test, y_test_pred)
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".format(
                loss, train_acc, test_acc))
    return nnet


def adam(nnet, X_train, y_train, minibatch_size, epoch, learning_rate, verbose=True,
         X_test=None, y_test=None):
    beta1 = 0.9
    beta2 = 0.999
    minibatches = get_minibatches(X_train, y_train, minibatch_size)
    for i in range(epoch):
        loss = 0
        velocity, cache = [], []
        for param_layer in nnet.params:
            p = [np.zeros_like(param) for param in list(param_layer)]
            velocity.append(p)
            cache.append(p)
        if verbose:
            print("Epoch {0}".format(i + 1))
        t = 1
        for X_mini, y_mini in minibatches:
            loss, grads = nnet.train_step(X_mini, y_mini)
            for c, v, param, grad, in zip(cache, velocity, nnet.params, reversed(grads)):
                for i in range(len(grad)):
                    c[i] = beta1 * c[i] + (1. - beta1) * grad[i]
                    v[i] = beta2 * v[i] + (1. - beta2) * (grad[i]**2)
                    mt = c[i] / (1. - beta1**(t))
                    vt = v[i] / (1. - beta2**(t))
                    param[i] += - learning_rate * mt / (np.sqrt(vt) + 1e-4)
            t += 1

        if verbose:
            train_acc = accuracy(y_train, nnet.predict(X_train))
            test_acc = accuracy(y_test, nnet.predict(X_test))
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".format(
                loss, train_acc, test_acc))
    return nnet
