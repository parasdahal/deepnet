import numpy as np
from deepnet.utils import softmax
from deepnet.layers import Conv, FullyConnected


def l2_regularization(layers, lam=0.001):
    reg_loss = 0.0
    for layer in layers:
        if hasattr(layer, 'W'):
            reg_loss += 0.5 * lam * np.sum(layer.W * layer.W)
    return reg_loss


def delta_l2_regularization(layers, grads, lam=0.001):
    for layer, grad in zip(layers, reversed(grads)):
        if hasattr(layer, 'W'):
            grad[0] += lam * layer.W
    return grads


def l1_regularization(layers, lam=0.001):
    reg_loss = 0.0
    for layer in layers:
        if hasattr(layer, 'W'):
            reg_loss += lam * np.sum(np.abs(layer.W))
    return reg_loss


def delta_l1_regularization(layers, grads, lam=0.001):
    for layer, grad in zip(layers, reversed(grads)):
        if hasattr(layer, 'W'):
            grad[0] += lam * layer.W / (np.abs(layer.W) + 1e-8)
    return grads


def SoftmaxLoss(X, y):
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m

    dx = p.copy()
    dx[range(m), y] -= 1
    dx /= m
    return loss, dx
