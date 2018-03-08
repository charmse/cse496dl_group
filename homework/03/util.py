import itertools as itr
import tensorflow as tf 
import numpy as np

EPS = 1e-10

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

def upscale_block(x, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, 3, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, padding='same')

def kl_divergence(p, q):
    """ inputs must be in (0, 1) """
    return p * tf.log(p/q) + (1-p) * tf.log((1-p)/(1-q))

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

def gaussian_encoder(inputs, latent_size):
    """inputs should be a tensor of images whose height and width are multiples of 4"""
    x = conv_block(inputs, 8, downscale=2)
    x = conv_block(x, 16, downscale=2)
    mean = tf.layers.dense(x, latent_size)
    log_scale = tf.layers.dense(x, latent_size)
    return mean, log_scale

def gaussian_sample(mean, log_scale):
    # noise is zero centered and std. dev. 1
    gaussian_noise = tf.random_normal(shape=tf.shape(mean))
    return mean + (tf.exp(log_scale) * gaussian_noise)

def upscale_block(x, scale=2, name="upscale_block"):
     """[Sub-Pixel Convolution](https://arxiv.org/abs/1609.05158) """
     n, w, h, c = x.get_shape().as_list()
     x = tf.layers.conv2d(x, c * scale ** 2, (3, 3), activation=tf.nn.relu, padding='same', name=name)
     output = tf.depth_to_space(x, scale)
     return output

def decoder(inputs, output_shape):
     """output_shape should be a length 3 iterable of ints"""
     h, w, c = output_shape
     initial_shape = [h // 4, w // 4, c]
     initial_size = reduce(mul, initial_shape)
     x = tf.layers.dense(inputs, initial_size // 32, name="decoder_dense")
     x = tf.reshape(x, [-1] + initial_shape)
     x = upscale_block(x, name="upscale_1")
     return upscale_block(x, name="upscale_2")

def std_gaussian_KL_divergence(mu, log_sigma):
    """Analytic KL distance between N(mu, e^log_sigma) and N(0, 1)"""
    sigma = tf.exp(log_sigma)
    return -0.5 * tf.reduce_sum(
        1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), 1)

def flatten(inputs):
    """
    Flattens a tensor along all non-batch dimensions.
    This is correctly a NOP if the input is already flat.
    """
    if len(shape(inputs)) == 2:
        return inputs
    else:
        size = inputs.get_shape().as_list()[1:]
        return tf.reshape(inputs, [-1, size])
    
def bernoulli_logp(alpha, sample):
    """Calculates log prob of sample under bernoulli distribution.
    
    Note: args must be in range [0,1]
    """
    alpha = flatten(alpha)
    sample = flatten(sample)
    re_x = tf.nn.relu(flatten(re_x))
    return tf.reduce_sum(x_in * tf.log(EPS + re_x) +
                         ((1 - x_in) * tf.log(EPS + 1 - re_x)), 1)

def discretized_logistic_logp(mean, logscale, sample, binsize=1 / 256.0):
    """Calculates log prob of sample under discretized logistic distribution."""
    scale = tf.exp(logscale)
    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
    logp = tf.log(
        tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + EPS)

    if logp.shape.ndims == 4:
        logp = tf.reduce_sum(logp, [1, 2, 3])
    elif logp.shape.ndims == 2:
        logp = tf.reduce_sum(logp, 1)
    return logp

def vae_loss(inputs, outputs, latent_mean, latent_log_scale, output_dist, output_log_scale=None):
    """Calculate the VAE loss (aka [ELBO](https://arxiv.org/abs/1312.6114))
    
    Args:
        - inputs: VAE input
        - outputs: VAE output
        - latent_mean: parameter of latent distribution
        - latent_log_scale: log of std. dev. of the latent distribution
        - output_dist: distribution parameterized by VAE output, must be in ['logistic', 'bernoulli']
        - output_log_scale: log scale parameter of the output dist if it's logistic, can be learnable
        
    Note: output_log_scale must be specified if output_dist is logistic
    """
    # Calculate reconstruction loss
    # Equal to minus the log likelihood of the input data under the VAE's output distribution
    if output_dist == 'bernoulli':
        outputs = tf.sigmoid(outputs)
        reconstruction_loss = bernoulli_joint_log_likelihood(outputs, inputs)
    elif output_dist == 'logistic':
        outputs = tf.clip_by_value(outputs, 1 / 512., 1 - 1 / 512.)
        reconstruction_loss = -discretized_logistic_logp(outputs, output_log_scale, inputs)
    else:
        print('Must specify an argument for output_dist in [bernoulli, logistic]')
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        
    # Calculate latent loss
    latent_loss = std_gaussian_KL_divergence(latent_mean, latent_log_scale)
    latent_loss = tf.reduce_mean(latent_loss)
    
    return reconstruction_loss, latent_loss