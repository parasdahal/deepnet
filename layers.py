import numpy as np
from im2col import *

class Conv():

    def __init__(self,X_dim,n_filter,h_filter,w_filter,stride,padding):

        self.d_X,self.h_X,self.w_X = X_dim

        self.n_filter,self.h_filter,self.w_filter = n_filter,h_filter,w_filter
        self.stride,self.padding = stride,padding

        self.W = np.random.randn(n_filter,self.d_X,h_filter,w_filter) / np.sqrt(n_filter/2.)
        self.b = np.zeros((self.n_filter,1))
        self.params = [self.W,self.b]

        self.h_output = ((self.h_X - h_filter + 2*padding)/ stride) + 1
        self.w_output = ((self.w_X - w_filter + 2*padding)/ stride) + 1
        

        if not self.h_output.is_integer() or not self.w_output.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_output,self.w_output  = int(self.h_output), int(self.w_output)
        self.out_dim = (self.n_filter,self.h_output,self.w_output)

    def forward(self,X):
        
        self.n_X = X.shape[0]

        self.X_col = im2col_indices(X,self.h_filter,self.w_filter,stride=self.stride,padding=self.padding)        
        W_row = self.W.reshape(self.n_filter,self.d_X*self.h_filter*self.w_filter)

        out = np.dot(W_row,self.X_col) + self.b

        out = out.reshape(self.n_filter,self.h_output,self.w_output,X.shape[0]).transpose(3,0,1,2)
        return out

    def backward(self,dout):

        dout_flat = dout.transpose(1,2,3,0).reshape(self.n_filter,-1)

        dW = np.dot(dout_flat,self.X_col.T)
        dW = dW.reshape(self.W.shape)

        db = np.sum(dout,axis=(0,2,3)).reshape(self.n_filter,-1)

        W_flat = self.W.reshape(self.n_filter,-1)

        dX_col = np.dot(W_flat.T,dout_flat)
        shape = (self.n_X,self.d_X,self.h_X,self.w_X)
        dX = col2im_indices(dX_col,shape,self.h_filter,self.w_filter,self.padding,self.stride)

        return dX, (dW, db)

class Maxpool():

    def __init__(self,X_dim,size,stride):

        self.d_X, self.h_X, self.w_X = X_dim
        
        self.params = []

        self.size = size
        self.stride = stride
        
        self.h_out = (self.h_X - size)/stride + 1
        self.w_out = (self.w_X - size)/stride + 1
        

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")
        
        self.h_out,self.w_out  = int(self.h_out), int(self.w_out)
        self.out_dim = (self.d_X,self.h_out,self.w_out)

    def forward(self,X):
        self.n_X = X.shape[0]
        X_reshaped = X.reshape(X.shape[0]*X.shape[1],1,X.shape[2],X.shape[3])

        self.X_col = im2col_indices(X_reshaped, self.size, self.size, padding = 0, stride = self.stride)
        
        self.max_indexes = np.argmax(self.X_col,axis=0)
        out = self.X_col[self.max_indexes,range(self.max_indexes.size)]

        out = out.reshape(self.h_out,self.w_out,self.n_X,self.d_X).transpose(2,3,0,1)
        return out

    def backward(self,dout):

        dX_col = np.zeros_like(self.X_col)
        # flatten the gradient
        dout_flat = dout.transpose(2,3,0,1).ravel()
        
        dX_col[self.max_indexes,range(self.max_indexes.size)] = dout_flat
        
        # get the original X_reshaped structure from col2im
        shape = (self.n_X*self.d_X,1,self.h_X,self.w_X)
        dX = col2im_indices(dX_col,shape,self.size,self.size,padding=0,stride=self.stride)
        dX = dX.reshape(self.n_X,self.d_X,self.h_X,self.w_X)
        return dX,()

class FullyConnected():

    def __init__(self,X_dim,out_size):

        self.d_X,self.h_X,self.w_X = X_dim
        self.W = np.random.rand(int(self.d_X*self.h_X*self.w_X),out_size)/np.sqrt(int(self.d_X*self.h_X*self.w_X)/2.)
        self.b = np.zeros((1,out_size))
        self.params = [self.W,self.b]

    def forward(self,X):
        self.n_X = X.shape[0]
        self.X = X.ravel().reshape(self.n_X,-1)
        out = np.dot(self.X,self.W) + self.b
        return out
    
    def backward(self,dout):
        
        dW = np.dot(self.X.T,dout)
        db = np.sum(dout,axis=0)
        dX = np.dot(dout,self.W.T).reshape(self.n_X,self.d_X,self.h_X,self.w_X)
        return dX,(dW,db)

class Batchnorm():

    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.gamma = np.ones((1,input_shape))
        self.beta = np.zeros((1,input_shape))
        self.params = [self.gamma,self.beta]

    def forward(self,X):
        self.X_flat = X.reshape(-1,self.input_shape)
        
        self.mu = np.mean(self.X_flat,axis=0)
        self.var = np.var(self.X_flat, axis=0)
        self.X_norm = (self.X_flat - self.mu)/np.sqrt(self.var + 1e-8)
        out = self.gamma * self.X_norm + self.beta
        
        return out.reshape(X.shape)

    def backward(self,dout):
        dout = dout.reshape(-1,self.input_shape)

        X_mu = self.X_flat - self.mu
        var_inv = 1./np.sqrt(self.var + 1e-8)
        
        dbeta = np.sum(dout,axis=0)
        dgamma = dout * self.X_norm

        dX_norm = dout * self.gamma
        dvar = np.sum(dX_norm * X_mu,axis=0) * -0.5 * (self.var + 1e-8)**(-3/2)
        dmu = np.sum(dX_norm * -var_inv ,axis=0) + dvar * 1/n_X * np.sum(-2.* X_mu, axis=0)
        dX = (dX_norm * var_inv) + (dmu / n_X) + (dvar * 2/n_X * X_mu)
        
        dX = dX.reshape(n_X,d_X,h_X,w_X)
        return dX, (dgamma, dbeta)

class Dropout():

    def __init__(self,prob=0.5):
        self.prob = prob
        self.params = []

    def forward(self,X):
        self.mask = np.random.binomial(1,self.prob,size=X.shape) / self.prob
        out = X * self.mask
        return out.reshape(X.shape)
    
    def backward(self,dout):
        dX = dout * self.mask
        return dX,()