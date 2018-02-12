import itertools as itr
import tensorflow as tf 
import numpy as np

def make(x):
    x_norm = x/255
    KEEP_PROB = 0.8
    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(x_norm, KEEP_PROB)
        hidden_1 = tf.layers.dense(dropped_input,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_1')
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        output = tf.layers.dense(dropped_hidden_1,
                                    10,
                                    name='output_layer')
    tf.identity(output, name='output')
    return output