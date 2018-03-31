import itertools as itr
import tensorflow as tf 
import numpy as np
import os
import sys
import model
import util
import ptb_reader

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/hackathon/08/ptbdata', 'directory where MNIST is located')
flags.DEFINE_string('train_dir', '', 'directory where training data is location')
flags.DEFINE_string('test_dir', '', '')
flags.DEFINE_string('save_dir', '', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 20, '')
flags.DEFINE_float('lr', 1e-4, '')
flags.DEFINE_integer('early_stop', 12, '')
flags.DEFINE_string('db', 'cifar-100', '')
flags.DEFINE_integer('epochs', 1, '')
flags.DEFINE_float('reg_coeff', 0.001, '')
flags.DEFINE_float('split', 0.90, '')
flags.DEFINE_integer('time_steps', 25, '')
flags.DEFINE_integer('vocab_size', 10000, '')
flags.DEFINE_integer('embedding_size', 100, '')
flags.DEFINE_integer('lstm_size', 200, '')
FLAGS = flags.FLAGS

def main(argv):

    # Set arguments  
    EPOCHS = FLAGS.epochs
    DATA_DIR = FLAGS.data_dir
    LEARNING_RATE = FLAGS.lr
    early_stop = FLAGS.early_stop
    BATCH_SIZE = FLAGS.batch_size
    reg_coeff = FLAGS.reg_coeff
    TIME_STEPS = FLAGS.time_steps
    VOCAB_SIZE = FLAGS.vocab_size
    EMBEDDING_SIZE = FLAGS.embedding_size
    LSTM_SIZE = FLAGS.lstm_size
    train_dir = FLAGS.train_dir
    test_dir = FLAGS.test_dir

    sys.path.append("/work/cse496dl/shared/hackathon/08")

    raw_data = ptb_reader.ptb_raw_data(DATA_DIR)
    train_data, valid_data, test_data, _ = raw_data
    train_input = PTBInput(train_data, BATCH_SIZE, TIME_STEPS, name="TrainInput")
    #print("The time distributed training data: " + str(train_input.input_data))
    #print("The similarly distributed targets: " + str(train_input.targets))

    # setup input and embedding
    embedding_matrix = tf.get_variable('embedding_matrix', dtype=tf.float32, shape=[VOCAB_SIZE, EMBEDDING_SIZE], trainable=True)
    word_embeddings = tf.nn.embedding_lookup(embedding_matrix, train_input.input_data)
    #print("The output of the word embedding: " + str(word_embeddings))

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)

    # Initial state of the LSTM memory.
    initial_state = lstm_cell.zero_state(BATCH_SIZE, tf.float32)
    
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, word_embeddings,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
    #print("The outputs over all timesteps: "+ str(outputs))
    #print("The final state of the LSTM layer: " + str(state))
    logits = tf.layers.dense(outputs, VOCAB_SIZE)


    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        train_input.targets,
        tf.ones([BATCH_SIZE, TIME_STEPS], dtype=tf.float32),
        average_across_timesteps=True,
        average_across_batch=True)

    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # start queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    # retrieve some data to look at
    examples = session.run([train_input.input_data, train_input.targets])
    # we can run the train op as usual
    _ = session.run(train_op)

    #print("Example input data:\n" + str(examples[0][1]))
    # print("Example target:\n" + str(examples[1][1]))

    for epoch in range(EPOCHS):
        _ = session.run(train_op)

    a = session.run([test_input.input_data])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, VOCAB_SIZE]))
    predict = tf.cast(tf.argmax(softmax_out, axis=1), tf.int32)

    a = session.run([test_input.input_data[0]])
    b = session.run([predict])

    session.run([test_input.targets[0]])
    pred = session.run(predict)
    pred
    session.run(test_input.targets[0])
    correct_prediction = tf.equal(pred, tf.reshape(test_input.targets, [-1]))
    session.run(correct_prediction[0])
    session.run(correct_prediction)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_val = session.run(accuracy)

    allfile = open('output/all_models_out.csv', 'a+')
    allfile.write(str(LSTM_SIZE) + ',' + str(VOCAB_SIZE) + ',' + str(EMBEDDING_SIZE) + ',' + str(EPOCHS) + ',' + str(BATCH_SIZE) + ',' + str(TIME_STEPS) + ',' + str(LEARNING_RATE) + ',' + str(accuracy_val) +"\n")
    allfile.close()



class PTBInput(object):
        """The input data.
        
        Code sourced from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
        """

        def __init__(self, data, batch_size, num_steps, name=None):
            self.batch_size = batch_size
            self.num_steps = num_steps
            self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
            self.input_data, self.targets = ptb_reader.ptb_producer(
                data, batch_size, num_steps, name=name)

if __name__ == "__main__":
    tf.app.run()