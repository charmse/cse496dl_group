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
        encoder_16 = downscale_block(x, filter=np.floor(x.get_shape().as_list()[3] * 1.25))
        encoder_8 = downscale_block(encoder_16, filter=np.floor(encoder_16.get_shape().as_list()[3] * 1.25))
        flatten_dim = np.prod(encoder_8.get_shape().as_list()[1:])
        flat = tf.reshape(encoder_8, [-1, flatten_dim])
        code_en = tf.layers.dense(flat, 96, activation=tf.nn.elu)
        code = tf.layers.dense(code_en, code_size, activation=tf.nn.elu)
        decoder_input = tf.identity(code, 'decoder_input')
        code_de = tf.layers.dense(decoder_input, 96, activation=tf.nn.elu)
        hidden_decoder = tf.layers.dense(code_de, 192, activation=tf.nn.elu)
        decoder_8 = tf.reshape(hidden_decoder, [-1, 8, 8, 3])
        decoder_8 = upscale_block(decoder_8)
        outputs = upscale_block(decoder_8)
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
