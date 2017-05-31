import numpy as np
from utils import load_mnist,load_cifar10
from layers import Conv,Maxpool,FullyConnected,Batchnorm,Dropout
from nonlinearity import ReLU
from solver import sgd,sgd_momentum
from nnet import NeuralNet
import sys

def make_mnist_cnn(X_dim,num_class):
    conv = Conv(X_dim,n_filter=5,h_filter=3,w_filter=3,stride=1,padding=1)
    relu_conv = ReLU()
    maxpool = Maxpool(conv.out_dim,size=2,stride=1)
    fc = FullyConnected(maxpool.out_dim,num_class)
    relu_fc = ReLU()
    return [conv,relu_conv,maxpool,fc,relu_fc]

def make_cifar10_cnn(X_dim,num_class):
    conv = Conv(X_dim,n_filter=5,h_filter=3,w_filter=3,stride=1,padding=1)
    relu = ReLU()
    maxpool = Maxpool(conv.out_dim,size=2,stride=1)
    conv2 = Conv(maxpool.out_dim,n_filter=5,h_filter=3,w_filter=3,stride=1,padding=1)
    relu2 = ReLU()
    maxpool2 = Maxpool(conv2.out_dim,size=2,stride=1)
    fc1 = FullyConnected(maxpool2.out_dim,num_class)
    return [conv,relu,maxpool,conv2,relu2,maxpool2,fc1]

if __name__ == "__main__":

    if sys.argv[1] == "mnist":
        
        training_set , test_set , _ = load_mnist(cnn=True,one_hot=False)
        X,y = training_set
        X_test,y_test = test_set
        X,y = X[0:1000,:,:,:],y[0:1000,]
        X_test,y_test = X_test[0:100,:,:,:],y[0:100,]
        mnist_dims = (1,28,28)
        cnn = NeuralNet( make_mnist_cnn(mnist_dims,num_class=10) )
        cnn = sgd_momentum(cnn,X,y,minibatch_size=35,epoch=50,learning_rate=0.1,\
                            X_test=X_test,y_test = y_test,nesterov=True)
    
    if sys.argv[1] == "cifar10":
        training_set , test_set = load_cifar10(num_training=1000,num_test=100)
        X,y = training_set
        X_test,y_test = test_set
        cifar10_dims = (3,32,32)
        cnn = NeuralNet( make_cifar10_cnn(cifar10_dims,num_class=10) )
        cnn = sgd_momentum(cnn,X,y,minibatch_size=10,epoch=100,learning_rate=0.1,X_test=X_test,y_test = y_test)
        # image = X[0].transpose(1,2,0)
        # import matplotlib.pyplot as plt
        # plt.imshow(np.uint8(image))
        # plt.show()