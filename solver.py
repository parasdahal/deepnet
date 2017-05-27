import numpy as np
from sklearn.utils import shuffle

def vanilla_update(params,grads,learning_rate=0.01):
    for param,grad in zip(params,reversed(grads)):
        for i in range(len(grad)):
            param[i] += - learning_rate * grad[i]

def get_minibatches(X,y,minibatch_size):
    m = X.shape[0]
    minibatches = []
    X,y = shuffle(X,y)
    for i in range (0,m,minibatch_size):
        X_batch = X[i:i+minibatch_size,:,:,:]
        y_batch = y[i:i+minibatch_size]
        minibatches.append((X_batch,y_batch))
    return minibatches

def sgd(nnet,X,y,minibatch_size,epoch,lr):
    minibatches = get_minibatches(X,y,minibatch_size)
    for i in range(epoch):
        X_mini, y_mini = minibatches[np.random.randint(0,len(minibatches))]
        print(X_mini.shape,y_mini.shape)
        out,loss = nnet.forward(X_mini,y_mini)
        grads = nnet.backward(out,y_mini)
        vanilla_update(nnet.params,grads,learning_rate = lr)
    return nnet
