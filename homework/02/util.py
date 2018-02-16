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

def onehot(data):
    data_final = np.zeros((data.shape[0],10))
    data_final[np.arange(data.shape[0]),data.astype(int)]=1
    return data_final

def extend(x, n):
    for i in range(len(x),n):
        x.append(x[len(x)-1])
    return x

def get_activators(strs):
    activs = []
    for s in strs:
        if s == "relu":
            activs.append(tf.nn.relu)
        elif s == "relu6":
            activs.append(tf.nn.relu6)
        elif s == "crelu":
            activs.append(tf.nn.crelu)
        elif s == "elu":
            activs.append(tf.nn.elu)
        elif s == "splus":
            activs.append(tf.nn.softplus)
        elif s == "ssign":
            activs.append(tf.nn.softsign)
        elif s == "sigmoid":
            activs.append(tf.nn.bias_add)
        elif s == "bias+":
            activs.append(tf.sigmoid)
        elif s == "tanh":
            activs.append(tf.tanh)
        else:
            activs.append(tf.nn.relu)
    return activs

def get_regularizers(types, vals):
    regs = []
    vals = extend(vals, len(types))
    for t,v in zip(types, vals):
        if t == "l1":
            regs.append(tf.contrib.layers.l1_regularizer(scale = float(v)))
        elif t == "l2":
            regs.append(tf.contrib.layers.l2_regularizer(scale = float(v)))
        elif t == "d":
            regs.append(float(v))
        else:
            regs.append(None) 
    return regs

def get_pools(pools):
    ret = []
    for p in pools:
        if bool(p):
            p = p.split('.')
            p = list(map(int, p))
            if len(p) == 1:
                p.append(p[0])
            ret.append(p)
        else:
            ret.append(None)
    return ret

def dense_block(inputs, neurons, activs, regs, block_num):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints of length n
    """

    block_name = "dense_block_"+str(block_num)

    #check to make sure filters is a list
    if(not isinstance(neurons, list)):
        return inputs
    n = len(neurons)
            
    #check activs 
    if(not isinstance(activs, list)):
        activs = extend([activs], n)
    elif(len(activs) != n):
        activs = extend(activs, n)
    #check regs
    if(not isinstance(regs, list)):
        regs = extend([regs], n)
    elif(len(regs) != n):
        regs = extend(regs, n)
    
    layers = []
    layers.append(inputs)
    i = 1
    layer_number = 1
    with tf.name_scope(block_name) as scope:
        for n,a,r in zip(neurons,activs,regs):
            layer_name = block_name + "_dense_layer_" + str(layer_number)
            if(isinstance(r, float)):
                layer = tf.layers.dense(layers[i-1], n, activation = a, name = layer_name)
                dropout_layer = tf.layers.dropout(layer, r,name = layer_name + "_dropout")
                layers.append(layer)
                layers.append(dropout_layer)
                i += 2
            else:
                layer = tf.layers.dense(layers[i-1], n, activation = a, kernel_regularizer = r, bias_regularizer = r, name = layer_name)
                layers.append(layer)
                i += 1
            layer_number += 1
            
    return layers[i-1]

def conv1d_block(inputs, filters, kernels, activs, regs, block_num):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints of length n
    """

    block_name = "conv_block_"+str(block_num)

    #check to make sure filters is a list
    if(not isinstance(filters, list)):
        return inputs
    n = len(filters)

    #check if kernels is list and is long enough
    if(not isinstance(kernels, list)):
        kernels = extend([kernels], n)
    elif(len(kernels) != n):
        kernels = extend(kernels, n)
            
    #check activs 
    if(not isinstance(activs, list)):
        activs = extend([activs], n)
    elif(len(activs) != n):
        activs = extend(activs, n)

    #check regs
    if(not isinstance(regs, list)):
        regs = extend([regs], n)
    elif(len(regs) != n):
        regs = extend(regs, n)

    layers = []
    layers.append(inputs)
    i = 1
    layer_number = 1
    with tf.name_scope(block_name) as scope:
        for f,k,a,r in zip(filters,kernels,activs,regs):
            layer_name = block_name + "_conv_layer_" + str(layer_number)
            layer = tf.layers.conv1d(layers[i-1], f, k, 1, activation = a, kernel_regularizer = r, padding='same', name = layer_name)
            pool = tf.layers.max_pooling2d(layer, 2, 2, padding='same')
            layers.append(layer)
            layers.append(pool)
            i += 2
            layer_number += 1

    return layers[i-1]

def conv2d_block(inputs, filters, kernels, activs, regs, pools, block_num):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints of length n
    """

    block_name = "conv_block_"+str(block_num)

    #check to make sure filters is a list
    if(not isinstance(filters, list)):
        return inputs
    n = len(filters)

    #check if kernels is list and is long enough
    if(not isinstance(kernels, list)):
        kernels = extend([kernels], n)
    elif(len(kernels) != n):
        kernels = extend(kernels, n)
            
    #check activs 
    if(not isinstance(activs, list)):
        activs = extend([activs], n)
    elif(len(activs) != n):
        activs = extend(activs, n)

    #check regs
    if(not isinstance(regs, list)):
        regs = extend([regs], n)
    elif(len(regs) != n):
        regs = extend(regs, n)

    #check pools
    if(not isinstance(pools, list)):
        pools = extend([pools], n)
    elif(len(pools) != n):
        pools = extend(pools, n)

    layers = []
    layers.append(inputs)
    i = 1
    layer_number = 1
    with tf.name_scope(block_name) as scope:
        for f,k,a,r,p in zip(filters,kernels,activs,regs,pools):
            layer_name = block_name + "_conv_layer_" + str(layer_number)
            if p == None:
                layer = tf.layers.conv2d(layers[i-1], f, k, 1, activation = a, kernel_regularizer = r, padding='same', name = layer_name)
                layers.append(layer)
                i += 1
            else:
                layer = tf.layers.conv2d(layers[i-1], f, k, 1, activation = a, kernel_regularizer = r, padding='same', name = layer_name)
                pool = tf.layers.max_pooling2d(layer, p[0], p[1], padding='same')
                layers.append(layer)
                layers.append(pool)
                i += 2
            layer_number += 1

    return layers[i-1]

def conv3d_block(inputs, filters, kernels, activs, regs, block_num):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints of length n
    """

    block_name = "conv_block_"+str(block_num)

    #check to make sure filters is a list
    if(not isinstance(filters, list)):
        return inputs
    n = len(filters)

    #check if kernels is list and is long enough
    if(not isinstance(kernels, list)):
        kernels = extend([kernels], n)
    elif(len(kernels) != n):
        kernels = extend(kernels, n)
            
    #check activs 
    if(not isinstance(activs, list)):
        activs = extend([activs], n)
    elif(len(activs) != n):
        activs = extend(activs, n)

    #check regs
    if(not isinstance(regs, list)):
        regs = extend([regs], n)
    elif(len(regs) != n):
        regs = extend(regs, n)

    layers = []
    layers.append(inputs)
    i = 1
    layer_number = 1
    with tf.name_scope(block_name) as scope:
        for f,k,a,r in zip(filters,kernels,activs,regs):
            layer_name = block_name + "_conv_layer_" + str(layer_number)
            layer = tf.layers.conv3d(layers[i-1], f, k, 1, activation = a, kernel_regularizer = r, padding='same', name = layer_name)
            pool = tf.layers.max_pooling2d(layer, 2, 2, padding='same')
            layers.append(layer)
            layers.append(pool)
            i += 2
            layer_number += 1

    return layers[i-1]

def sep_conv2d_block(inputs, filters, kernels, activs, regs, block_num):
    """
    Args:
        - inputs: 4D tensor of shape NHWC
        - filters: iterable of ints of length 3
    """

    block_name = "sep_conv_block_"+str(block_num)

    #check to make sure filters is a list
    if(not isinstance(filters, list)):
        return inputs
    n = len(filters)

    #check if kernels is list and is long enough
    if(not isinstance(kernels, list)):
        kernels = extend([kernels], n)
    elif(len(kernels) != n):
        kernels = extend(kernels, n)
            
    #check activs 
    if(not isinstance(activs, list)):
        activs = extend([activs], n)
    elif(len(activs) != n):
        activs = extend(activs, n)

    #check regs
    if(not isinstance(regs, list)):
        regs = extend([regs], n)
    elif(len(regs) != n):
        regs = extend(regs, n)
    
    layers = []
    layers.append(inputs)
    i = 1
    layer_number = 1
    with tf.name_scope(block_name) as scope:
        for f,k,a,r in zip(filters,kernels,activs,regs):
            layer_name = block_name + "_sep_conv_layer_" + str(layer_number)
            layer = tf.layers.separable_conv2d(layers[i-1], f, k, 1, activation = a, pointwise_regularizer = r, depthwise_regularizer = r, padding='same', name = layer_name)
            pool = tf.layers.max_pooling2d(layer, 2, 2, padding='same')
            layers.append(layer)
            layers.append(pool)
            i += 2
            layer_number += 1

    return layers[i-1]