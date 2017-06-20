import numpy as np
from deepnet.nnet import RNN
from deepnet.solver import sgd_rnn


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
        return X, y, vocab_size, char_to_idx, idx_to_char


if __name__ == "__main__":

    X, y, vocab_size, char_to_idx, idx_to_char = text_to_inputs('data/Rnn.txt')
    rnn = RNN(vocab_size,vocab_size,char_to_idx,idx_to_char)
    rnn = sgd_rnn(rnn,X,y,10,10,0.1)



    
