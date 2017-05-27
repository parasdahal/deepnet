import numpy as np
from layers import Conv,Maxpool,FullyConnected,Batchnorm,Dropout
from nonlinearity import ReLU
from loss import CrossEntropy
from utils import load_mnist,accuracy,softmax
from solver import sgd,vanilla_update

class NeuralNet:

    def __init__(self,layers,loss_func=CrossEntropy):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)
        self.loss_func = loss_func

    def forward(self,X,y=None):
        volume = X
        for layer in self.layers:
            volume=layer.forward(volume)
        if y is not None:
            loss = self.loss_func.func(volume,y)
            return volume,loss
        return volume

    def predict(self,X):
        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)

    def backward(self,out,y):
        dout = self.loss_func.delta(out,y)
        grads = []
        for layer in reversed(self.layers):
            dout,grad = layer.backward(dout)
            grads.append(grad)
        return grads

if __name__ == "__main__":

    def make_cnn(X_dim,num_class):
        conv = Conv(X_dim,n_filter=3,h_filter=3,w_filter=3,stride=1,padding=1)
        relu_conv = ReLU()
        maxpool = Maxpool(conv.out_dim,size=2,stride=1)
        conv2 = Conv(maxpool.out_dim,n_filter=3,h_filter=2,w_filter=2,stride=1,padding=1)
        relu_conv2 = ReLU()
        maxpool2 = Maxpool(conv2.out_dim,size=2,stride=1)
        fc = FullyConnected(maxpool.out_dim,num_class)
        relu_fc = ReLU()
        return [conv,relu_conv,maxpool,conv2,relu_conv2,maxpool2,fc,relu_fc]
    
    
    training_set , _ , _ = load_mnist(cnn=True,one_hot=False)
    X,y = training_set
    X,y = X[0:1000,:,:,:],y[0:1000,]
    image_dims = (1,28,28)
    cnn = NeuralNet( make_cnn(image_dims,num_class=10) )
    cnn = sgd(cnn,X,y,minibatch_size=50,epoch=5,learning_rate=0.1)