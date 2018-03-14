import itertools as itr
import tensorflow as tf 
import numpy as np
import util
import sys

def upscale_block(x, filter=3, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, filter, 5, strides=(scale, scale), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))

def downscale_block(x, filter=3, scale=2):
    return tf.layers.conv2d(x, filter, 5, strides=scale, padding='same',activation = tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))

def autoencoder_network(x, code_size, model):
    if(model == 'default'):
        encoder_16 = downscale_block(x)
        encoder_8 = downscale_block(encoder_16)
        flatten_dim = np.prod(encoder_8.get_shape().as_list()[1:])
        flat = tf.reshape(encoder_8, [-1, flatten_dim])
        code_en = tf.layers.dense(flat, 96, activation=tf.nn.relu)
        code = tf.layers.dense(code_en, code_size, activation=tf.nn.relu, name='encoder_output')
        decoder_input = tf.identity(code, name='decoder_input')
        code_de = tf.layers.dense(decoder_input, 96, activation=tf.nn.relu)
        hidden_decoder = tf.layers.dense(code_de, 192, activation=tf.nn.elu)
        decoder_8 = tf.reshape(hidden_decoder, [-1, 8, 8, 3])
        decoder_16 = upscale_block(decoder_8)
        outputs = upscale_block(decoder_16)
        tf.identity(outputs, name='decoder_output')
    elif(model == '3'):
        encoder_16 = downscale_block(x)
        flatten_dim = np.prod(encoder_16.get_shape().as_list()[1:])
        flat = tf.reshape(encoder_16, [-1, flatten_dim])
        code = tf.layers.dense(flat, 768, activation=tf.nn.elu, name='encoder_output')
        decoder_input = tf.identity(code, name='decoder_input')
        decoder_16 = tf.reshape(decoder_input, [-1, 16, 16, 3])
        outputs = upscale_block(decoder_16)
        tf.identity(outputs, name='decoder_output')
    elif(model == '4'):
        encoder_16 = downscale_block(x)
        flatten_dim = np.prod(encoder_16.get_shape().as_list()[1:])
        flat = tf.reshape(encoder_16, [-1, flatten_dim])
        code_en = tf.layers.dense(flat, 768, activation=tf.nn.relu)
        code = tf.layers.dense(code_en, code_size, activation=tf.nn.relu, name='encoder_output')
        decoder_input = tf.identity(code, name='decoder_input')
        hidden_decoder = tf.layers.dense(decoder_input, 768, activation=tf.nn.relu)
        decoder_16 = tf.reshape(hidden_decoder, [-1, 16, 16, 3])
        outputs = upscale_block(decoder_16)
        tf.identity(outputs, name='decoder_output')
    else:
        print("Error: Auto Encoder Model Not Found!")
        sys.exit(1)
    return code, outputs, model
