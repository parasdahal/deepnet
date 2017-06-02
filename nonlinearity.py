import numpy as np

class ReLU():
    def __init__(self):
        self.params = []
    
    def forward(self,X):
        self.X = X
        return np.maximum(X,0)
    
    def backward(self,dout):
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX,[]

class Sigmoid():

    def forward(self,X):
        self.X = X
        return 1.0/(1.0+np.exp(X))

    def backward(self,dout):
        dX = dout * self.X * (1-self.X)
        return dX,()