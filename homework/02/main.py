import itertools as itr
import tensorflow as tf 
import numpy as np
import os
import sys
import model
import util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '$WORK/cse496dl/cse496dl_group/homework/02/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS

def main(argv):
    # Set arguments
    struct = sys.argv[1]
    learning_rate = float(sys.argv[2])
    if len(sys.argv) > 3:
        save_dir = sys.argv[3]
    else:
        save_dir = FLAGS.save_dir

    # load training data
    train_images_1 = np.load(FLAGS.data_dir + 'train_x_1.npy')
    train_images_2 = np.load(FLAGS.data_dir + 'train_x_2.npy')
    train_images_3 = np.load(FLAGS.data_dir + 'train_x_3.npy')
    train_images_4 = np.load(FLAGS.data_dir + 'train_x_4.npy')
    train_labels_1 = util.onehot(np.load(FLAGS.data_dir + 'train_y_1.npy'))
    train_labels_2 = util.onehot(np.load(FLAGS.data_dir + 'train_y_2.npy'))
    train_labels_3 = util.onehot(np.load(FLAGS.data_dir + 'train_y_3.npy'))
    train_labels_4 = util.onehot(np.load(FLAGS.data_dir + 'train_y_4.npy'))

    # load testing data
    test_images_1 = np.load(FLAGS.data_dir + 'test_x_1.npy')
    test_images_2 = np.load(FLAGS.data_dir + 'test_x_2.npy')
    test_images_3 = np.load(FLAGS.data_dir + 'test_x_3.npy')
    test_images_4 = np.load(FLAGS.data_dir + 'test_x_4.npy')
    test_labels_1 = util.onehot(np.load(FLAGS.data_dir + 'test_y_1.npy'))
    test_labels_2 = util.onehot(np.load(FLAGS.data_dir + 'test_y_2.npy'))
    test_labels_3 = util.onehot(np.load(FLAGS.data_dir + 'test_y_3.npy'))
    test_labels_4 = util.onehot(np.load(FLAGS.data_dir + 'test_y_4.npy'))

    # split into train and validate
    train_images_1, valid_images_1, train_labels_1, valid_labels_1 = util.split_data(train_images_1, train_labels_1, .90)
    train_images_2, valid_images_2, train_labels_2, valid_labels_2 = util.split_data(train_images_2, train_labels_2, .90)
    train_images_3, valid_images_3, train_labels_3, valid_labels_3 = util.split_data(train_images_3, train_labels_3, .90)
    train_images_4, valid_images_4, train_labels_4, valid_labels_4 = util.split_data(train_images_4, train_labels_4, .90)

    #Create list of 
    train_images = [train_images_1, train_images_2, train_images_3, train_images_4]
    train_labels = [train_labels_1, train_labels_2, train_labels_3, train_labels_4]
    valid_images = [valid_images_1, valid_images_2, valid_images_3, valid_images_4]
    valid_labels = [valid_labels_1, valid_labels_2, valid_labels_3, valid_labels_4]
    test_images = [test_images_1, test_images_2, test_images_3, test_images_4]
    test_labels = [test_labels_1, test_labels_2, test_labels_3, test_labels_4]

    # specify the network
    x = tf.placeholder(tf.float32, [None, 128, 128, 1], name='input_placeholder')
    output = model.make(x,struct)
    tf.identity(output, name='output')

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 7], name='label')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    red_mean = tf.reduce_mean(cross_entropy)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)) , tf.float32))

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)
    saver = tf.train.Saver()

    #Open file to write to
    myfile = open(save_dir + 'output/model_' + struct + '_out.txt', 'w+')

    #Create lists to collect best models
    best_epochs = []
    best_train_ces = []
    best_validation_ces = []
    best_accuracies = []
    best_conf_mxs = []
    i = 1
    for train_images, train_labels, valid_images, valid_labels, test_images, test_labels in zip(train_images, train_labels, valid_images, valid_labels, test_images, test_labels):
        train_num_examples = train_images.shape[0]
        valid_num_examples = valid_images.shape[0]
        test_num_examples = test_images.shape[0]
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
                for i in range(valid_num_examples // batch_size):
                    batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = valid_labels[i*batch_size:(i+1)*batch_size, :]       
                    validate_ce, conf_matrix, accuracy = session.run([red_mean, confusion_matrix_op, accuracy_op], {x: batch_xs, y: batch_ys})
                    ce_vals.append(validate_ce)
                    conf_mxs.append(conf_matrix)
                    accuracy_vals.append(accuracy)
                avg_validation_ce = sum(ce_vals) / len(ce_vals)
                avg_accuracy = sum(accuracy_vals) / len(accuracy_vals)

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
                    best_model = saver.save(session, os.path.join(save_dir + "models/", "emodb_homework_2-0_" + struct + "_" + str(i)))
                    count = 0
                else:
                    count += 1

                if count > 12:
                    break

            myfile.write("BEST VALIDATION CROSS-ENTROPY" +
            "\n-----------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")
            
            #Collect best's
            best_epochs.append(best_epoch)
            best_train_ces.append(best_train_ce)
            best_validation_ces.append(best_validation_ces)
            best_accuracies.append(best_accuracy)
            best_conf_mxs.append(best_conf_mx)
            i += 1

    myfile.write("AVERAGE BEST VALIDATION CROSS-ENTROPY" +
    "\n-----------------------------" +
    "\nAVERAGE EPOCH: " + str(sum(best_epochs)/len(best_epochs)) +
    "\nAVERAGE TRAIN LOSS: " + str(sum(best_train_ces)/len(best_train_ces)) +
    "\nAVERAGE VALIDATION LOSS: " + str(sum(best_validation_ces)/len(best_validation_ces)) +
    "\nAVERAGE ACCURACY: " + str(sum(best_accuracies)/len(best_accuracies)) +
    "\nAVERAGE CONFUSION MATRIX: \n" + str(sum(best_conf_mxs)/len(best_conf_mxs)) +
    "\n------------------------------------------\n")

    myfile.close()

if __name__ == "__main__":
    tf.app.run()