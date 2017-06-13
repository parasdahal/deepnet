import numpy as np
from deepnet.utils import softmax

v_size = 64
h_size = 64

model = dict(
    Wxh = np.random.rand(v_size, h_size) / np.sqrt(v_size / 2),
    Whh = np.random.rand(h_size, h_size) / np.sqrt(h_size / 2),
    Why = np.random.rand(h_size, v_size) / np.sqrt(h_size / 2),
    bh = np.zeros((1, v_size)),
    by = np.zeros((1, h_size))
)


def forward(X, h):

    # input to one hot
    X_onehot = np.zeros((1, v_size))
    X_onehot[X] = 1

    h_prev = h.copy()
    # calculate hidden step with tanh
    h = np.tanh(X @ model['Wxh'] + h_prev @ model['Whh'] + model['bh'])

    # fully connected forward step
    out = X @ model['Why'] + by

    cache = (X_onehot, h_prev)

    return out, h, cache


def backward(out, y, dh_next, cache):

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


def text_to_inputs(path):
    """
    Converts the given text into X and y vectors
    X : contains the index of all the characters in the text vocab
    y : y[i] contains the index of next character for X[i] in the text vocab
    """
    with open(path) as f:
        txt = f.read()
        X, y = [], []

        char_to_idx = {char: i for i, char in enumerate(set(txt))}
        idx_to_char = {i: char for i, char in enumerate(set(txt))}
        X = np.array([char_to_idx[i] for i in txt])
        y = [char_to_idx[i] for i in txt[1:]]
        y.append(char_to_idx['.'])
        y = np.array(y)

        vocab_size = len(char_to_idx)
        return dict(X=X, y=y, vocab_size=vocab_size,
                    char_to_idx=char_to_idx, idx_to_char=idx_to_char)


if __name__ == "__main__":

    print(text_to_inputs('data/rnn.txt'))
