import numpy as np

def cross_entropy(output,true_labels):
    cost = np.sum( np.nan_to_num(-true_labels*np.log(output) - (1-true_labels)*np.log(1-output) ))
    return cost
