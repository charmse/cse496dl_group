import itertools as itr
import tensorflow as tf 
import numpy as np
import util
import sys

def upscale_block(x, filter=3, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, filter, 3, strides=(scale, scale), padding='same', activation=tf.nn.elu)

def downscale_block(x, filter=3, scale=2):
    return tf.layers.conv2d(x, filter, 3, strides=scale, padding='same',activation = tf.nn.elu)

def autoencoder_network(x, code_size, model):
    
    if(model == 'default'):
        #x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='encoder_input')
        flatten_dim = np.prod(x.get_shape().as_list()[1:])
        flat = tf.reshape(x, [-1, flatten_dim])
        code = tf.layers.dense(flat, flat.get_shape.as_list()[1], activation=tf.nn.elu)
        decoder_input = tf.identity(code, 'decoder_input')
        outputs = tf.reshape(decoder_input, [-1, 32, 32, 3])
        tf.identity(outputs, name='decoder_output')
    elif(model == 'maxcompression'):
        en_conv_64 = tf.layers.conv2d(x, 64, 3, strides=2, padding='same',activation = tf.nn.elu)
        en_conv_128 = tf.layers.conv2d(en_conv_64, 128, 3, strides= 2, padding='same', activation=tf.nn.elu)
        flatten_dim = np.prod(en_conv_128.get_shape().as_list()[1:])
        flat = tf.reshape(en_conv_128, [-1, flatten_dim])
        en_dense_1 = tf.layers.dense(flat, 100000, activation=tf.nn.elu)
        en_dense_2 = tf.layers.dense(en_dense_1, 10000, activation=tf.nn.elu)
        code = tf.layers.dense(en_dense_2, 1000, activation=tf.nn.elu, name='encoder_output')
        de_dense_1 = tf.layers.dense(flat, 10000, activation=tf.nn.elu, name='decoder_input')
        de_dense_2 = tf.layers.dense(en_dense_2, 100000, activation=tf.nn.elu)
        de_trans_128 = tf.layers.conv2d_transpose(tf.reshape(de_dense_2, [-1, 4, 4, 3]), 128, 3, strides=(2, 2), padding='same', activation=tf.nn.elu)
        de_trans_64 = tf.layers.conv2d_transpose(de_trans_128, 64, 3, strides=(2, 2), padding='same', activation=tf.nn.elu)
        outputs = tf.layers.conv2d_transpose(de_trans_64, 3, 3, strides=(2,2), padding='same', activation=tf.nn.elu)
        tf.identity(outputs, name='decoder_output')
    else:
        print("Error: Auto Encoder Model Not Found!")
        sys.exit(1)
    return code, outputs, model
