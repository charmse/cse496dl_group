import itertools as itr
import tensorflow as tf 
import numpy as np

def split_data(data, labels, proportion):
    """                                                                                                                                                                                                                                                                                                 
    """
    size = data.shape[0]
    np.random.seed(42)
    indicies = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[indicies[:split_idx]], data[indicies[split_idx:]], labels[indicies[:split_idx]], labels[indicies[split_idx:]]

def transform_labels(labels):
    new_labels = []
    for l in labels:
        t_label = np.zeros(10)
        t_label[(int(l)-1)] = 1.
        new_labels.append(t_label)
    return np.array(new_labels)
