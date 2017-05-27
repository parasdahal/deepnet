import numpy as np
from utils import softmax


class CrossEntropy:

    @staticmethod
    def func(X,y):
        """
        Calculate cross entropy loss for distribution X
        given empirical distribution y
        
        Parameters
        ----------
        X : ndarray of size num_training x num_class
        y : ndarray of size num_training x 1
        """    
        m = y.shape[0]
        p = softmax(X)
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss

    @staticmethod
    def delta(X,y):
        m = y.shape[0]
        grad = softmax(X)
        grad[range(m),y] -= 1
        grad = grad/m
        return grad