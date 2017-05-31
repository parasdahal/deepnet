import numpy as np
from utils import softmax
from layers import Conv,FullyConnected

def l2_reg(W,lam):
    return 0.5 * lam * np.sum(W * W)

def dl2_reg(W,lam):
    return lam * W

def regularization(layers,reg=l2_reg,lam=0.001):
    reg_loss = 0.0
    weight_layers = (Conv,FullyConnected)
    for layer in layers:
        if isinstance(layer,weight_layers):
            reg_loss += reg(layer.W,lam)
    return reg_loss

def delta_regularization(layers,reg=dl2_reg,lam=0.001):
    weight_layers = (Conv,FullyConnected)
    for layer in layers:
        if isinstance(layer,weight_layers):
            layer.W = reg(layer.W,lam)


class CrossEntropy:

    @staticmethod
    def func(X,y):
        """
        X : ndarray of size num_training x num_class
        y : ndarray of size num_training x 1
        """    
        m = y.shape[0]
        p = softmax(X)
        p[p == 0] = 1e-8
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss

    @staticmethod
    def delta(X,y):
        m = y.shape[0]
        grad = softmax(X)
        grad[range(m),y] -= 1
        grad = grad/m
        return grad