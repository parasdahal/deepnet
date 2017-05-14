import numpy as np
from im2col import *

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