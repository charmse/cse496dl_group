tf.reset_default_graph()
test_input = Input(batch_size=20, num_steps=20, data=test_data)
m_test = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocabulary,
              num_layers=1)
saver = tf.train.Saver()
output = m_test.output
logits = m_test.logits
init_state = m_test.init_state
state = m_test.state
loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            test_input.targets,
            tf.ones([batch_size, num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
cost = tf.reduce_sum(loss)

        # get the prediction accuracy
softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocabulary]))
predict = tf.cast(tf.argmax(softmax_out, axis=1), tf.int32)
correct_prediction = tf.equal(predict, tf.reshape(test_input.targets, [-1]))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((1, 2, m_test.batch_size, m_test.hidden_size))
        # restore the trained model
        saver.restore(sess, "/work/cse496dl/vsunkara/04/homework/04/models/homework_4-0-final")
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
                print("input data:")
                print(" ".join(data_input_string))
                print("True values (1st line) vs predicted values (2nd line):")
                print(" ".join(true_vals_string))
                print(" ".join(pred_string))
                #acc, current_state = sess.run([accuracy, m_test.state], feed_dict={m_test.init_state: current_state})
                if batch >= acc_check_thresh:
                    accuracy_final += acc
                    print(accuracy_final)
        print("Average accuracy: {}".format(accuracy_final / (num_acc_batches-acc_check_thresh)))
        # close threads
        coord.request_stop()
        coord.join(threads)