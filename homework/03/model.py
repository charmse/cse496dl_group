import itertools as itr
import tensorflow as tf 
import numpy as np
import util
import sys

def upscale_block(x, filter=3, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, filter, 5, strides=(scale, scale), padding='same', activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))

def downscale_block(x, filter=3, scale=2):
    return tf.layers.conv2d(x, filter, 5, strides=scale, padding='same',activation = tf.nn.elu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))

def autoencoder_network(x, code_size, model):
    if(model == 'default'):
        #x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='encoder_input')
        encoder_16 = downscale_block(x, filter=np.floor(x.get_shape().as_list()[3] * 1.25))
        #encoder_8 = downscale_block(encoder_16, filter=np.floor(encoder_16.get_shape().as_list()[3] * 1.25))
        #encoder_4 = downscale_block(encoder_8, filter = np.floor(encoder_8.get_shape().as_list()[3] * 1.25))
        flatten_dim = np.prod(encoder_16.get_shape().as_list()[1:])
        flat = tf.reshape(encoder_16, [-1, flatten_dim])
        #code_en = tf.layers.dense(flat, 150, activation=tf.nn.elu)
        code = tf.layers.dense(flat, code_size, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.))
        #code = flat
        tf.identity(code,name='encoder_output')
        #code = tf.cast(code,tf.int32)
        code_de = tf.placeholder(tf.float32, [None, code_size], name='decoder_input')
        #code_de = tf.cast(code,tf.float32)
        code_de = code
        #code_de1 = tf.layers.dense(code_de,150,activation=tf.nn.elu)
        hidden_decoder = tf.layers.dense(code_de,768, activation=tf.nn.elu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.))
        decoder_16 = tf.reshape(hidden_decoder, [-1, 16, 16, 3])
        #decoder_8 = upscale_block(decoder_4)
        #decoder_16 = upscale_block(decoder_8)
        outputs = upscale_block(decoder_16)
        tf.identity(outputs, name='decoder_output')
    else:
        print("Error: Auto Encoder Model Not Found!")
        sys.exit(1)
    return code, outputs, model
