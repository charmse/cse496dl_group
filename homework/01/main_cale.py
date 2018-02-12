import itertools as itr
import tensorflow as tf 
import numpy as np
import os
import model_cale
import util_cale

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '/work/soh/charms/cse496dl/homework/01/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS

def main(argv):
    # load data
    train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
    train_labels = util_cale.transform_labels(np.load(FLAGS.data_dir + 'fmnist_train_labels.npy'))

    # split into train and validate
  
    train_images, validation_images, train_labels, validation_labels = util_cale.split_data(train_images, train_labels, .90)

    validation_num_examples = validation_images.shape[0]
    train_num_examples = train_images.shape[0]
    
    # specify the network
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    output = model_cale.make(x)

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    red_mean = tf.reduce_mean(cross_entropy)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)) , tf.float32))

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training
        batch_size = FLAGS.batch_size
        best_validation_ce = float("inf")
        count = 0
        for epoch in range(FLAGS.max_epoch_num):

            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]       
                _, train_ce = session.run([train_op, red_mean], {x: batch_xs, y: batch_ys})
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)

            accuracy_vals = []
            ce_vals = []
            conf_mxs = []
            for i in range(validation_num_examples // batch_size):
                batch_xs = validation_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = validation_labels[i*batch_size:(i+1)*batch_size, :]       
                validate_ce, conf_matrix, accuracy = session.run([red_mean, confusion_matrix_op, accuracy_op], {x: batch_xs, y: batch_ys})
                ce_vals.append(validate_ce)
                conf_mxs.append(conf_matrix)
                accuracy_vals.append(accuracy)
            avg_validation_ce = sum(ce_vals) / len(ce_vals)
            avg_accuracy = sum(accuracy_vals) / len(accuracy_vals)

            with open('/work/soh/charms/cse496dl/homework/01/model_out.txt', 'a') as myfile:
                myfile.write("Epoch: " + str(epoch) +
                "\nTrain loss: " + str(avg_train_ce) +
                "\nValidation loss: " + str(avg_validation_ce) +
                "\nAccuracy: " + str(avg_accuracy) +
                "\n------------------------------\n")

            if avg_validation_ce < best_validation_ce:
                best_validation_ce = avg_validation_ce
                best_epoch = epoch
                best_train_ce = avg_train_ce
                best_conf_mx = sum(conf_mxs)
                best_accuracy = avg_accuracy
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0"))
                count = 0
            else:
                count += 1

            if count > 12:
                break

        with open('/work/soh/charms/cse496dl/homework/01/model_out.txt', 'a') as myfile:
            myfile.write("BEST VALIDATION CROSS-ENTROPY" +
                            "\n-----------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")

if __name__ == "__main__":
    tf.app.run()
