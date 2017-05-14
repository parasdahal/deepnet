import numpy as np
from im2col import *

class ConvLayer():

    def __init__(self,X_shape,num_filter,h_filter,w_filter,stride,padding):
        
        self.num_image,self.c_image,self.h_image,self.w_image = X_shape
        self.num_filter = num_filter
        self.h_filter = h_filter
        self.w_filter = w_filter
        self.stride = stride
        self.padding = padding
        
        self.W = np.random.randn(num_filter,self.c_image,h_filter,w_filter)
        self.b = np.zeros((self.num_filter,1))

        self.h_output = ((self.h_image - h_filter + 2*padding)/ stride) + 1
        self.w_output = ((self.w_image - w_filter + 2*padding)/ stride) + 1

        if not h_output.is_integer() or not w_output.is_integer():
            raise Exception("The given parameters are invalid for generating output volume!")

    def forward(self,X):
        
        self.X_col = im2col_indices(X,self.h_filter,self.w_filter,stride=self.stride,padding=self.padding)
        W_row = self.W.reshape(self.num_filter,self.c_image*self.h_filter*self.w_filter)
        out = np.dot(W_row,self.X_col) + self.b
        out = out.reshape(self.num_filter,self.h_output,self.w_output,self.num_image).transpose(3,0,1,2)
        return out

    def backward(self,dout):
        
        dout_flat = dout.transpose(1,2,3,0).reshape(self.num_filter,-1)
        
        dW = np.dot(dout_flat,self.X_col.T)
        dW = dW.reshape(self.W.shape)

        db = np.sum(dout,axis=(0,2,3)).reshape(self.num_filter,-1)

        W_flat = self.W.reshape(self.num_filter,-1)
        dX_col = np.dot(W_flat.T,dout_flat)
        X_shape = (self.num_image,self.c_image,self.h_image,self.w_image)
        dX = col2im_indices(dX_col,X_shape,self.h_filter,self.w_filter,self.padding,self.stride)
        return dX, dW, db