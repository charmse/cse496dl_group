import numpy as np
import os

def split_data(data, labels, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array of data, to be split along the first axis
        - labels: numpy array of the labels
        - proportion: a float less than 1  
        
    Returns:
        In order,
        - Validation set data
        - Training set data
        - Validation set labels
        - Training set labels
    """
    size = data.shape[0]
    np.random.seed(42)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return (data[s[:split_idx]], data[s[split_idx:]], labels[s[:split_idx]], labels[s[split_idx:]])

def transform_labels(labels):
    new_labels = []
    for l in labels:
        t_label = np.zeros(10)
        t_label[(int(l)-1)] = 1.
        new_labels.append(t_label)
    return np.array(new_labels)
