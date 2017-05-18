import numpy as np

class relu():
    
    def forward(self,X):
        self.X = X
        return np.maximum(X,0)
    
    def backward(self,dout):
        dX = dout.copy()
        # create a boolean map of all the elements less than 0 in the input from forward prop
        filter = self.X <= 0
        # replace elements less than 0 in actual input with 0 in dout
        dX[filter] = 0
        return dX

class sigmoid():

    def forward(self,X):
        self.X = X
        return 1.0/(1.0+np.exp(X))

    def backward(self,dout):
        dX = dout * self.X * (1-self.X)
        return dX