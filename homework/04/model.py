import itertools as itr
import tensorflow as tf 
import numpy as np
import util
import sys
import os

class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        # create the word embeddings
        embedding = tf.get_variable('embedding_matrix', dtype=tf.float32, shape=[vocab_size, self.hidden_size], trainable=True)
            #embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size]))
        inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # set up the state storage / extraction
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        # create an LSTM cell to be unrolled
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        if num_layers >= 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        self.output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)
        # reshape to (batch_size * num_steps, hidden_size)
        self.logits = tf.layers.dense(self.output, vocab_size)
