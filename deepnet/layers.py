import numpy as np
from deepnet.im2col import *


class Conv():

    def __init__(self, X_dim, n_filter, h_filter, w_filter, stride, padding):

        self.d_X, self.h_X, self.w_X = X_dim

        self.n_filter, self.h_filter, self.w_filter = n_filter, h_filter, w_filter
        self.stride, self.padding = stride, padding

        self.W = np.random.randn(
            n_filter, self.d_X, h_filter, w_filter) / np.sqrt(n_filter / 2.)
        self.b = np.zeros((self.n_filter, 1))
        self.params = [self.W, self.b]

        self.h_out = (self.h_X - h_filter + 2 * padding) / stride + 1
        self.w_out = (self.w_X - w_filter + 2 * padding) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.n_filter, self.h_out, self.w_out)

    def forward(self, X):

        self.n_X = X.shape[0]

        self.X_col = im2col_indices(
            X, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding)
        W_row = self.W.reshape(self.n_filter, -1)

        out = W_row @ self.X_col + self.b
        out = out.reshape(self.n_filter, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):

        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)

        dW = dout_flat @ self.X_col.T
        dW = dW.reshape(self.W.shape)

        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.n_filter, -1)

        W_flat = self.W.reshape(self.n_filter, -1)

        dX_col = W_flat.T @ dout_flat
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        dX = col2im_indices(dX_col, shape, self.h_filter,
                            self.w_filter, self.padding, self.stride)

        return dX, [dW, db]


class Maxpool():

    def __init__(self, X_dim, size, stride):

        self.d_X, self.h_X, self.w_X = X_dim

        self.params = []

        self.size = size
        self.stride = stride

        self.h_out = (self.h_X - size) / stride + 1
        self.w_out = (self.w_X - size) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.d_X, self.h_out, self.w_out)

    def forward(self, X):
        self.n_X = X.shape[0]
        X_reshaped = X.reshape(
            X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[3])

        self.X_col = im2col_indices(
            X_reshaped, self.size, self.size, padding=0, stride=self.stride)

        self.max_indexes = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_indexes, range(self.max_indexes.size)]

        out = out.reshape(self.h_out, self.w_out, self.n_X,
                          self.d_X).transpose(2, 3, 0, 1)
        return out

    def backward(self, dout):

        dX_col = np.zeros_like(self.X_col)
        # flatten the gradient
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        dX_col[self.max_indexes, range(self.max_indexes.size)] = dout_flat

        # get the original X_reshaped structure from col2im
        shape = (self.n_X * self.d_X, 1, self.h_X, self.w_X)
        dX = col2im_indices(dX_col, shape, self.size,
                            self.size, padding=0, stride=self.stride)
        dX = dX.reshape(self.n_X, self.d_X, self.h_X, self.w_X)
        return dX, []


class Flatten():

    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return out

    def backward(self, dout):
        out = dout.reshape(self.X_shape)
        return out, ()


class FullyConnected():

    def __init__(self, in_size, out_size):

        self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size / 2.)
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):
        self.X = X
        out = self.X @ self.W + self.b
        return out

    def backward(self, dout):
        dW = self.X.T @ dout
        db = np.sum(dout, axis=0)
        dX = dout @ self.W.T
        return dX, [dW, db]


class Batchnorm():

    def __init__(self, X_dim):
        self.d_X, self.h_X, self.w_X = X_dim
        self.gamma = np.ones((1, int(np.prod(X_dim))))
        self.beta = np.zeros((1, int(np.prod(X_dim))))
        self.params = [self.gamma, self.beta]

    def forward(self, X):
        self.n_X = X.shape[0]
        self.X_shape = X.shape

        self.X_flat = X.ravel().reshape(self.n_X, -1)
        self.mu = np.mean(self.X_flat, axis=0)
        self.var = np.var(self.X_flat, axis=0)
        self.X_norm = (self.X_flat - self.mu) / np.sqrt(self.var + 1e-8)
        out = self.gamma * self.X_norm + self.beta

        return out.reshape(self.X_shape)

    def backward(self, dout):

        dout = dout.ravel().reshape(dout.shape[0], -1)
        X_mu = self.X_flat - self.mu
        var_inv = 1. / np.sqrt(self.var + 1e-8)

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * self.X_norm, axis=0)

        dX_norm = dout * self.gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * - \
            0.5 * (self.var + 1e-8)**(-3 / 2)
        dmu = np.sum(dX_norm * -var_inv, axis=0) + dvar * \
            1 / self.n_X * np.sum(-2. * X_mu, axis=0)
        dX = (dX_norm * var_inv) + (dmu / self.n_X) + \
            (dvar * 2 / self.n_X * X_mu)

        dX = dX.reshape(self.X_shape)
        return dX, [dgamma, dbeta]


class Dropout():

    def __init__(self, prob=0.5):
        self.prob = prob
        self.params = []

    def forward(self, X):
        self.mask = np.random.binomial(1, self.prob, size=X.shape) / self.prob
        out = X * self.mask
        return out.reshape(X.shape)

    def backward(self, dout):
        dX = dout * self.mask
        return dX, []


class ReLU():
    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dout):
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX, []


class sigmoid():
    def __init__(self):
        self.params = []

    def forward(self, X):
        out = 1.0 / (1.0 + np.exp(X))
        self.out = out
        return out

    def backward(self, dout):
        dX = dout * self.out * (1 - self.out)
        return dX, []


class tanh():
    def __init__(self):
        self.params = []

    def forward(self, X):
        out = np.tanh(X)
        self.out = out
        return out

    def backward(self, dout):
        dX = dout * (1 - self.out**2)
        return dX, []
