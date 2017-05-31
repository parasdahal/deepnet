import numpy as np
from loss import CrossEntropy,regularization,delta_regularization
from utils import accuracy,softmax

from layers import Conv,Maxpool,FullyConnected,Batchnorm,Dropout
from nonlinearity import ReLU
from solver import sgd,vanilla_update

class NeuralNet:

    def __init__(self,layers,loss_func=CrossEntropy):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)
        self.loss_func = loss_func

    def forward(self,X,y=None):
        volume = X
        for layer in self.layers:
            volume=layer.forward(volume)
        if y is not None:
            loss = self.loss_func.func(volume,y)
            # reg_loss = regularization(self.layers)
            return volume,loss
        return volume

    def predict(self,X):
        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)

    def backward(self,out,y):
        dout = self.loss_func.delta(out,y)
        grads = []
        for layer in reversed(self.layers):
            dout,grad = layer.backward(dout)
            # delta_regularization(self.layers)
            grads.append(grad)
        return grads