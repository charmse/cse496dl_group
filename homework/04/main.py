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
flags.DEFINE_integer('num_steps', 20, '')
flags.DEFINE_integer('vocab_size', 10000, '')
flags.DEFINE_integer('embedding_size', 100, '')
flags.DEFINE_integer('lstm_size', 200, '')
flags.DEFINE_integer('lstm_layers', 1, '')
flags.DEFINE_integer('k', 1, '')
flags.DEFINE_boolean('train', True, '')
FLAGS = flags.FLAGS

def main(argv):

    # Set arguments  
    EPOCHS = FLAGS.epochs
    DATA_DIR = FLAGS.data_dir
    LEARNING_RATE = FLAGS.lr
    BATCH_SIZE = FLAGS.batch_size
    NUM_STEPS = FLAGS.num_steps
    VOCAB_SIZE = FLAGS.vocab_size
    EMBEDDING_SIZE = FLAGS.embedding_size
    LSTM_SIZE = FLAGS.lstm_size
    LSTM_LAYERS = FLAGS.lstm_layers
    k = FLAGS.k
    TRAIN = FLAGS.train

    sys.path.append("/work/cse496dl/shared/hackathon/08")

    train_data, valid_data, test_data, VOCAB_SIZE, reversed_dictionary = util.ptb_raw_data(DATA_DIR)
    training_input = util.Input(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, data=train_data)
    m = model.Model(training_input, is_training=True, hidden_size=LSTM_SIZE, vocab_size=VOCAB_SIZE,
            num_layers=LSTM_LAYERS)
    output = m.output
    logits = m.logits
    init_state = m.init_state
    state = m.state
    loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            training_input.targets,
            tf.ones([BATCH_SIZE, NUM_STEPS], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

    # Update the cost
    cost = tf.reduce_sum(loss)

        # get the prediction accuracy
    softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, VOCAB_SIZE]))
    predict = tf.cast(tf.argmax(softmax_out, axis=1), tf.int32)
    correct_prediction = tf.equal(predict, tf.reshape(training_input.targets, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    #self.learning_rate = tf.Variable(0.0, trainable=False)
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss)

    print_iter =50
    init_op = tf.global_variables_initializer()
    #orig_decay = lr_decay
    with tf.Session() as sess:
        # start threads
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        for epoch in range(EPOCHS):

            current_state = np.zeros((LSTM_LAYERS, 2, BATCH_SIZE, LSTM_SIZE))

            for step in range(training_input.epoch_size):
                # cost, _ = sess.run([m.cost, m.optimizer])
                if step % print_iter != 0:
                    cost1, _, current_state = sess.run([cost, train_op, state],
                                                    feed_dict={init_state: current_state})
                else:
                    #seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    #curr_time = dt.datetime.now()
                    cost1, _, current_state, acc = sess.run([cost, train_op, state, accuracy],
                                                        feed_dict={init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}".format(epoch,
                            step, cost1, acc))

            # save a model checkpoint
            saver.save(sess, os.path.join("models/", "homework_4-0"))
        # do a final save
        saver.save(sess, os.path.join("models/", "homework_4-0"))
        # close threads
        coord.request_stop()
        coord.join(threads)

    tf.reset_default_graph()
    test_input = util.Input(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, data=test_data)
    m_test = model.Model(test_input, is_training=False, hidden_size=LSTM_SIZE, vocab_size=VOCAB_SIZE,
                num_layers=LSTM_LAYERS)
    saver = tf.train.Saver()
    output = m_test.output
    logits = m_test.logits
    init_state = m_test.init_state
    state = m_test.state
    loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                test_input.targets,
                tf.ones([BATCH_SIZE, NUM_STEPS], dtype=tf.float32),
                average_across_timesteps=False,
                average_across_batch=True)

            # Update the cost
    cost = tf.reduce_sum(loss)

            # get the prediction accuracy
    softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, VOCAB_SIZE]))
    predict = tf.cast(tf.argmax(softmax_out, axis=1), tf.int32)
    correct_prediction = tf.equal(predict, tf.reshape(test_input.targets, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
            # start threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            current_state = np.zeros((LSTM_LAYERS, 2, m_test.batch_size, m_test.hidden_size))
            # restore the trained model
            saver.restore(sess, "models/homework_4-0")
            # get an average accuracy over num_acc_batches
            num_acc_batches = 30
            check_batch_idx = 25
            acc_check_thresh = 5
            accuracy_final = 0
            for batch in range(num_acc_batches):
                    current_state = np.zeros((1, 2, m_test.batch_size, m_test.hidden_size))
                    data_input,true_vals, pred, current_state, acc = sess.run([m_test.input_obj.input_data,m_test.input_obj.targets, predict, m_test.state, accuracy],
                                                                feed_dict={m_test.init_state: current_state})
                    pred_string = [reversed_dictionary[x] for x in pred[:m_test.num_steps]]
                    true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
                    data_input_string = [reversed_dictionary[x] for x in data_input[0]]

                    if batch >= acc_check_thresh:
                        accuracy_final += acc

            allfile = open('output/all_models_out.csv', 'a+')
            allfile.write(str(LSTM_SIZE) + ',' + str(k) + ',' + str(VOCAB_SIZE) + ',' + str(EMBEDDING_SIZE) + ',' + str(EPOCHS) + ',' + str(BATCH_SIZE) + ',' + str(TIME_STEPS) + ',' + str(LEARNING_RATE) + ',' + str(accuracy_final / (num_acc_batches-acc_check_thresh)) +"\n")
            allfile.close()

            # close threads
            coord.request_stop()
            coord.join(threads)

    # raw_data = ptb_reader.ptb_raw_data(DATA_DIR)
    # train_data, valid_data, test_data, _ = raw_data
    # train_input = PTBInput(train_data, BATCH_SIZE, TIME_STEPS, name="TrainInput")
    # test_input = PTBInput(test_data,BATCH_SIZE,TIME_STEPS,name="TestInput")

    # # setup input and embedding
    # embedding_matrix = tf.get_variable('embedding_matrix', dtype=tf.float32, shape=[VOCAB_SIZE, EMBEDDING_SIZE], trainable=True)
    # word_embeddings = tf.nn.embedding_lookup(embedding_matrix, train_input.input_data)

    # lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)

    # # Initial state of the LSTM memory.
    # initial_state = lstm_cell.zero_state(BATCH_SIZE, tf.float32)
    
    # outputs, state = tf.nn.dynamic_rnn(lstm_cell, word_embeddings,
    #                                initial_state=initial_state,
    #                                dtype=tf.float32)

    # logits = tf.layers.dense(outputs, VOCAB_SIZE)

    # loss = tf.contrib.seq2seq.sequence_loss(
    #     logits,
    #     train_input.targets,
    #     tf.ones([BATCH_SIZE, TIME_STEPS], dtype=tf.float32),
    #     average_across_timesteps=True,
    #     average_across_batch=True)

    # optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
    # train_op = optimizer.minimize(loss)
    # softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, VOCAB_SIZE]))
    # predict = tf.cast(tf.argmax(softmax_out, axis=1), tf.int32)

    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())

    #     # start queue runners
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=session, coord=coord)

    #     # retrieve some data to look at
    #     examples = session.run([train_input.input_data, train_input.targets])

    #     #train
    #     for epoch in range(EPOCHS):
    #         for i in range(train_input.input_data.shape[0] // BATCH_SIZE):
    #             _ = session.run(train_op)

    #     a = session.run([test_input.input_data[0]])
    #     b = session.run([predict])

    #     correct_prediction = tf.equal(predict, tf.reshape(test_input.targets, [-1]))
    #     session.run(correct_prediction[0])
    #     session.run(correct_prediction)

    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     accuracy_val = session.run(accuracy)

    # allfile = open('output/all_models_out.csv', 'a+')
    # allfile.write(str(LSTM_SIZE) + ',' + str(k) + ',' + str(VOCAB_SIZE) + ',' + str(EMBEDDING_SIZE) + ',' + str(EPOCHS) + ',' + str(BATCH_SIZE) + ',' + str(TIME_STEPS) + ',' + str(LEARNING_RATE) + ',' + str(accuracy_val) +"\n")
    # allfile.close()



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
