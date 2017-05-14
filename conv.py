import numpy as np
from im2col import *

class Conv():

    def __init__(self,X_shape,num_filter,h_filter,w_filter,stride,padding):
        
        # shape of input volume
        self.num_image,self.c_image,self.h_image,self.w_image = X_shape
        
        # parameters that define output volume
        self.num_filter,self.h_filter,self.w_filter = num_filter,h_filter,w_filter
        self.stride,self.padding = stride,padding
        
        # create and initialize weights and bias for the layer
        self.W = np.random.randn(num_filter,self.c_image,h_filter,w_filter)
        self.b = np.zeros((self.num_filter,1))

        # calculate the shape of output volume
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