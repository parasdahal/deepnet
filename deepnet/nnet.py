import numpy as np
from deepnet.loss import SoftmaxLoss, l2_regularization, delta_l2_regularization
from deepnet.utils import accuracy, softmax


class NeuralNet:

    def __init__(self, layers, loss_func=SoftmaxLoss):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)
        self.loss_func = loss_func

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        grads = []
        for layer in reversed(self.layers):
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self, X, y):
        out = self.forward(X)
        loss, dout = self.loss_func(out, y)
        loss += l2_regularization(self.layers)
        grads = self.backward(dout)
        grads = delta_l2_regularization(self.layers, grads)
        return loss, grads

    def predict(self, X):
        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)


class RNN:

    def __init__(self, vocab_size, h_size, char_to_idx, idx_to_char):
        self.vocab_size = vocab_size
        self.h_size = h_size
        self.char_to_idx = char_to_idx
        self.idx_to_char
        self.model = dict(
            Wxh=np.random.rand(v_size, h_size) / np.sqrt(v_size / 2),
            Whh=np.random.rand(h_size, h_size) / np.sqrt(h_size / 2),
            Why=np.random.rand(h_size, v_size) / np.sqrt(h_size / 2),
            bh=np.zeros((1, v_size)),
            by=np.zeros((1, h_size))
        )

    def _forward(X, h):
        # input to one hot
        X_onehot = np.zeros((1, self.vocab_size))
        X_onehot[X] = 1

        h_prev = h.copy()
        # calculate hidden step with tanh
        h = np.tanh(X @ self.model['Wxh'] + h_prev @
                    self.model['Whh'] + self.model['bh'])

        # fully connected forward step
        y = X @ model['Why'] + model['by']

        cache = (X_onehot, h_prev)
        return y, h, cache

    def _backward(out, y, dh_next, cache):

        X_onehot, h_prev = cache

        # gradient of output from froward step
        dout = softmax(out)
        dout[range(len(y)), y] -= 1
        # fully connected backward step
        dWhy = X_onehot.T @ dout
        dby = np.sum(dWhy, axis=0).reshape(1, -1)
        dh = dout @ self.dWhy.T
        # gradient through tanh
        dh = dout * (1 - out**2)
        # add up gradient from previous gradient
        dh += dh_next
        # hidden state
        dbh = dh
        dWhh = h_prev.T @ dh
        dWxh = X_onehot.T @ dh
        dh_next = dh @ Whh.T

        grads = dict(Wxh=dWxh, Whh=dWhh, Why=dWhy, bh=dbh, by=dby)

        return grads, dh_next

    def train_step(self,X_train, y_train, h):
        ys, caches = [], []
        total_loss = 0
        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        # forward pass and store values for bptt
        for x, y in zip(X_train, y_train):
            y, h, cache = self._forward(x, h)
            loss, _ = SoftmaxLoss(x, y)
            total_loss += loss
            ys.append(y)
            caches.append(cache)

        total_loss /= X_train.shape[0]

        # backprop through time
        dh_next = np.zeros((1, h_size))
        for t in reversed(range(len(X_train))):
            grad, dh_next = self._backward(
                ys[t], y_train[t], dh_next, caches[t])
            # sum up the gradients for each time step
            for k in grads.keys():
                grads[k] += grad[k]

        # clip vanishing/exploding gradients
        for k, v in grads.items():
            grads[k] = np.clip(v, -5.0, 5.0)

        return grads, loss, h
