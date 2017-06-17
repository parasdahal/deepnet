# deepnet

Implementations of CNNs, RNNs and cool new techniques in deep learning

Note: deepnet is a work in progress and things will be added gradually. It is not intended for production, use it to learn and study implementations of latest and greatest in deep learning.

## What does it have?

**Network Architecture**
1. Convolutional net
2. Feed forward net
3. Recurrent net (LSTM/GRU coming soon)

**Optimization Algorithms**
1. SGD
2. SGD with momentum
3. Nesterov Accelerated Gradient
4. Adagrad
5. RMSprop
6. Adam

**Regularization**
1. Dropout
2. L1 and L2 Regularization

**Cool Techniques**

1. BatchNorm
2. Xavier Weight Initialization

**Nonlinearities**
1. ReLU
2. Sigmoid
3. tanh


## Usage

1. ```virtualenv .env``` ; create a virtual environment
2. ```source .env/bin/activate``` ; activate the virtual environment
3. ```pip install -r requirements.txt``` ; Install dependencies
4. ```python run_cnn.py {mnist|cifar10}``` ; mnist for shallow cnn and cifar10 for deep cnn