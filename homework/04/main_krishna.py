import numpy as np
import tensorflow as tf
import collections
import sys
import os
import model
import util
# these ones let us draw images in our notebook

def main(argv):
    sys.path.append("/work/cse496dl/shared/hackathon/08")
    batch_size = 20
    num_steps =20
    data_path = "/work/cse496dl/shared/hackathon/08/ptbdata"
    train_data, valid_data, test_data, vocabulary, reversed_dictionary = ptb_raw_data(data_path)
    training_input = Input(batch_size=20, num_steps=20, data=train_data)
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocabulary,
              num_layers=1)
    output = m.output
    logits = m.logits
    init_state = m.init_state
    state = m.state
    loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            training_input.targets,
            tf.ones([batch_size, num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
    cost = tf.reduce_sum(loss)

        # get the prediction accuracy
    softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocabulary]))
    predict = tf.cast(tf.argmax(softmax_out, axis=1), tf.int32)
    correct_prediction = tf.equal(predict, tf.reshape(training_input.targets, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        #self.learning_rate = tf.Variable(0.0, trainable=False)
    LEARNING_RATE = 1e-4
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss)
        #tvars = tf.trainable_variables()
        #grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #self.train_op = optimizer.apply_gradients(
        #    zip(grads, tvars),
        #    global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        #self.new_lr = tf.placeholder(tf.float32, shape=[])
        #self.lr_update = tf.assign(self.learning_rate, self.new_lr)
    print_iter =50
    init_op = tf.global_variables_initializer()
    #orig_decay = lr_decay
    with tf.Session() as sess:
        # start threads
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        for epoch in range(1):
            #new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            #m.assign_lr(sess, learning_rate * new_lr_decay)
            # m.assign_lr(sess, learning_rate)
            # print(m.learning_rate.eval(), new_lr_decay)
            current_state = np.zeros((1, 2, batch_size, 650))
            #curr_time = dt.datetime.now()
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
            saver.save(sess, os.path.join("/work/cse496dl/vsunkara/04/homework/04/" + "models/", "homework_4-0"))
        # do a final save
        saver.save(sess, os.path.join("/work/cse496dl/vsunkara/04/homework/04/" + "models/", "homework_4-0"))
        # close threads
        coord.request_stop()
        coord.join(threads)

        


        
        

if __name__ == "__main__":
    tf.app.run()

