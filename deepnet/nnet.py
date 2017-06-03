import numpy as np
from deepnet.loss import SoftmaxLoss,l2_regularization,delta_l2_regularization
from deepnet.utils import accuracy,softmax

class NeuralNet:

    def __init__(self,layers,loss_func=SoftmaxLoss):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)
        self.loss_func = loss_func

    def forward(self,X):
        for layer in self.layers:
            X=layer.forward(X)
        return X

    def backward(self,dout):
        grads = []
        for layer in reversed(self.layers):
            dout,grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self,X,y):
        out = self.forward(X)
        loss,dout = self.loss_func(out,y)
        loss += l2_regularization(self.layers)
        grads = self.backward(dout)
        grads = delta_l2_regularization(self.layers,grads)
        return loss,grads
    
    def predict(self,X):
        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)
