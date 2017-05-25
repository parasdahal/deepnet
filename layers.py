import numpy as np
from im2col import *

class Conv():

    def __init__(self,X_shape,num_filter,h_filter,w_filter,stride,padding):

        self.num_image,self.c_image,self.h_image,self.w_image = X_shape

        self.num_filter,self.h_filter,self.w_filter = num_filter,h_filter,w_filter
        self.stride,self.padding = stride,padding

        self.W = np.random.randn(num_filter,self.c_image,h_filter,w_filter) / np.sqrt(num_filter/2.)
        self.b = np.zeros((self.num_filter,1))

        self.h_output = ((self.h_image - h_filter + 2*padding)/ stride) + 1
        self.w_output = ((self.w_image - w_filter + 2*padding)/ stride) + 1

        if not h_output.is_integer() or not w_output.is_integer():
            raise Exception("Invalid dimensions!")

    def forward(self,X):

        # convert input to columns of receptive fields
        self.X_col = im2col_indices(X,self.h_filter,self.w_filter,stride=self.stride,padding=self.padding)

        # flatten each filters to multiply with each receptive field
        # resultant matrix of each row is a flattened filter
        W_row = self.W.reshape(self.num_filter,self.c_image*self.h_filter*self.w_filter)

        # multiply each filter to each receptive field and add the bias term
        # resultant matrix is of size (num_filter x num_receptive_fields)
        # this matrix has dot product of every filter with every field, which is convolution
        out = np.dot(W_row,self.X_col) + self.b

        # unpack the convolved region
        # then transpose to give the output shape (num_image x num_filter x h_output x w_output )
        out = out.reshape(self.num_filter,self.h_output,self.w_output,self.num_image).transpose(3,0,1,2)
        return out

    def backward(self,dout):

        # we convert the gradient from previous layer to a flat
        # dout is of shape (num_image x num_filter x h_output x w_output)
        # transpose so that we flat dout into (num_filter * flattened image)
        dout_flat = dout.transpose(1,2,3,0).reshape(self.num_filter,-1)

        # multiply the gradient from prev layer with local gradient that is the input to the current layer
        # reshape the resultant to the shape of the weight matrix
        dW = np.dot(dout_flat,self.X_col.T)
        dW = dW.reshape(self.W.shape)

        # accumulate the gradient to get bias
        db = np.sum(dout,axis=(0,2,3)).reshape(self.num_filter,-1)

        # flatten the each filter in the weight matrix
        W_flat = self.W.reshape(self.num_filter,-1)

        # perform convolution with gradient from previous layer
        # but with spatially flipped filters along both the axes
        dX_col = np.dot(W_flat.T,dout_flat)

        X_shape = (self.num_image,self.c_image,self.h_image,self.w_image)
        # reshape the input gradient to the shape of input volume
        dX = col2im_indices(dX_col,X_shape,self.h_filter,self.w_filter,self.padding,self.stride)

        return dX, dW, db

class MaxPool():

    def __init__(self,X_shape,size,stride):
        # shape of input volume
        self.n_X , self.c_X, self.h_X, self.w_X = X_shape
        
        # parameters that define output volume
        self.size = size
        self.stride = stride
        
        # compute the output volume shape
        self.h_out = (self.h_X - size)/stride + 1
        self.w_out = (self.w_X - size)/stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

    def forward(self,X):
        # convert the channels into images so that each of them is converted to columns by im2col
        X_reshaped = X.reshape(self.n_X*self.c_X,1,self.h_X,self.w_X)

        self.X_col = im2col_indices(X_reshaped, self.size, self.size, padding = 0, stride = self.stride)
        
        # get the max value of each column i.e. each receptive field
        self.max_indexes = np.argmax(self.X_col,axis=0)
        # index out the max from each receptive field along all the columns
        out = self.X_col[self.max_indexes,range(self.max_indexes.size)]
        # reshape it to the size of the output volume
        out = out.reshape(self.h_out,self.w_out,self.n_X,self.c_X)
        out = out.transpose(2,3,0,1)
        return out

    def backward(self,dout):

        dX_col = np.zeros_like(self.X_col)
        # flatten the gradient
        dout_flat = dout.transpose(2,3,0,1).ravel()
        # fill the max indexes from the forward prop with the gradients
        dX_col[self.max_indexes,range(self.max_indexes.size)] = dout_flat
        # get the original X_reshaped structure from col2im
        dX = col2im_indices(dX_col,(self.n_X*self.c_X,1,self.h_X,self.w_X),self.size,self.size,padding=0,stride=self.stride)
        # from X_reshaped structure convert to X structure
        dX = dX.reshape(self.n_X , self.c_X, self.h_X, self.w_X)
        return dX

class FullyConnected():

    def __init__(self,X_shape,out_size):
        
        n_X,c_X,h_X,w_X = X_shape
        # the weight matrix will be flattened image x size of the output layer
        self.W = np.random.rand(c_X*h_X*w_X,out_size)/np.sqrt(c_X*h_X*w_X/2.)
        self.b = np.zeros((1,out_size))

    def forward(self,X):
        # flatten the volume and create a matrix of num_X x flattened image
        self.X = X.ravel().reshape(X.shape[0],-1)
        out = np.dot(self.X,self.W) + self.b
        # output shape will be num_X x size of the output layer
        return out
    
    def backward(self,dout):
        
        dW = np.dot(self.X.T,dout)
        db = np.sum(dout,axis=0)
        dX = np.dot(dout,self.W.T)
        
        return dX,dW,db

class BatchNorm():

    def __init__(self,X_shape):
        self.X_shape = X_shape
        n_X,c_X,h_X,w_X = self.X_shape
        self.gamma = np.ones((1,c_X*h_X*w_X)) 
        self.beta = np.zeros((1,c_X*h_X*w_X)) 

    def forward(self,X):
        n_X,c_X,h_X,w_X = self.X_shape
        self.X_flat = X.reshape(n_X,c_X*h_X*w_X)
        
        self.mu = np.mean(self.X_flat,axis=0)
        self.var = np.var(self.X_flat, axis=0)
        self.X_norm = (self.X_flat - self.mu)/np.sqrt(self.var + 1e-8)
        out = self.gamma * self.X_norm + self.beta
        
        return out.reshape(X.shape)

    def backward(self,dout):
        n_X,c_X,h_X,w_X = self.X_shape
        dout = dout.reshape(n_X,c_X*h_X*w_X)

        X_mu = self.X_flat - self.mu
        var_inv = 1./np.sqrt(self.var + 1e-8)
        
        dbeta = np.sum(dout,axis=0)
        dgamma = dout * self.X_norm

        dX_norm = dout * self.gamma
        dvar = np.sum(dX_norm * X_mu,axis=0) * -0.5 * (self.var + 1e-8)**(-3/2)
        dmu = np.sum(dX_norm * -var_inv ,axis=0) + dvar * 1/n_X * np.sum(-2.* X_mu, axis=0)
        dX = (dX_norm * var_inv) + (dmu / n_X) + (dvar * 2/n_X * X_mu)
        
        dX = dX.reshape(n_X,c_X,h_X,w_X)
        return dX, dgamma, dbeta

class Dropout():

    def __init__(self,X_shape,prob=0.5):
        self.X_shape = X_shape
        n_X,c_X,h_X,w_X = X_shape
        self.mask_shape = (n_X,c_X*h_X*w_X)
        self.prob = prob

    def forward(self,X):
        X_flat = X.reshape(self.mask_shape)
        self.mask = np.random.binomial(1,self.prob,size=self.mask_shape) / self.prob
        out = X_flat * self.mask
        return out.reshape(self.X_shape)
    
    def backward(self,dout):
        dout = dout.reshape(self.mask_shape)
        dX = dout * self.mask
        dX = dX.reshape(self.X_shape)
        return dX

