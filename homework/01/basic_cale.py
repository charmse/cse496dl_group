import itertools as itr
import tensorflow as tf 
import numpy as np
import os


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '/work/soh/charms/cse496dl/homework/01/basic/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS

def split_data(data, labels, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    
    Args:
        - data: numpy array of data, to be split along the first axis
        - labels: numpy array of the labels
        - proportion: a float less than 1  
        
    Returns:
        In order,
        - Validation set data
        - Training set data
        - Validation set labels
        - Training set labels
    """
    size = data.shape[0]
    np.random.seed(42)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return (data[s[:split_idx]], data[s[split_idx:]], labels[s[:split_idx]], labels[s[split_idx:]])

def transform_labels(labels):
    new_labels = []
    for l in labels:
        t_label = np.zeros(10)
        t_label[(int(l)-1)] = 1.
        new_labels.append(t_label)
    return np.array(new_labels)


def main(argv):
    # load data
    train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
    train_labels = transform_labels(np.load(FLAGS.data_dir + 'fmnist_train_labels.npy'))

    # split into train and validate
  
    sp_data = split_data(train_images, train_labels, .90)
    validation_images = sp_data[0]
    train_images = sp_data[1]
    validation_labels = sp_data[2]
    train_labels = sp_data[3]

    validation_num_examples = validation_images.shape[0]
    train_num_examples = train_images.shape[0]
    
    with open('/work/soh/charms/cse496dl/homework/01/basic/model_out_500__500_do_l2.txt', "w+") as myfile:
        # specify the network
        input_placeholder = tf.placeholder(tf.float32, [None, 784], name='data')
        input_norm = input_placeholder/255
        KEEP_PROB = 0.5

        dropped_input = tf.layers.dropout(input_norm, KEEP_PROB)
        hidden_1 = tf.layers.dense(dropped_input,
                                    500,
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                    name='hidden_layer_1')
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        hidden_2 = tf.layers.dense(dropped_hidden_1,
                                    500,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                    bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                    activation=tf.nn.relu,
                                    name='hidden_layer_2')
        dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
        output = tf.layers.dense(dropped_hidden_2,
                                    10,
                                    name='output_layer')
        tf.identity(output, name='model_output')

        # define classification loss
        y = tf.placeholder(tf.float32, [None, 10], name='label')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
        red_mean = tf.reduce_mean(cross_entropy)
        confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)) , tf.float32))


        # set up training and saving functionality
        global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)
        validate_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # run training
            batch_size = FLAGS.batch_size
            best_validation_ce = float("inf")
            best_accuracy_ = 0.0
            count = 0
            for epoch in range(FLAGS.max_epoch_num):

                # run gradient steps and report mean loss on train data
                ce_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]       
                    _, train_ce = session.run([train_op, red_mean], {input_placeholder: batch_xs, y: batch_ys})
                    ce_vals.append(train_ce)
                avg_train_ce = sum(ce_vals) / len(ce_vals)

                accuracy_vals = []
                ce_vals = []
                conf_mxs = []
                for i in range(validation_num_examples // batch_size):
                    batch_xs = validation_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = validation_labels[i*batch_size:(i+1)*batch_size, :]       
                    validate_ce, conf_matrix, accuracy = session.run([red_mean, confusion_matrix_op, accuracy_op], {input_placeholder: batch_xs, y: batch_ys})
                    ce_vals.append(validate_ce)
                    conf_mxs.append(conf_matrix)
                    accuracy_vals.append(accuracy)
                avg_validation_ce = sum(ce_vals) / len(ce_vals)
                avg_accuracy = sum(accuracy_vals) / len(accuracy_vals)

                file.write("Epoch: " + str(epoch) +
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
                    best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_val"))
                    count = 0

                if avg_accuracy > best_accuracy_:
                    best_validation_ce_ = avg_validation_ce
                    best_epoch_ = epoch
                    best_train_ce_ = avg_train_ce
                    best_conf_mx_ = sum(conf_mxs)
                    best_accuracy_ = avg_accuracy
                    best_model_ = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_acu"))
                    count = 0

            file.write("BEST VALIDATION CROSS-ENTROPY" +
                            "\n-----------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: " + str(best_conf_mx) +
            "\n------------------------------------------\n")

            file.write("BEST ACCURACY" +
                            "\n--------------" +
            "EPOCH: " + str(best_epoch_) +
            "\nTRAIN LOSS: " + str(best_train_ce_) +
            "\nVALIDATION LOSS: " + str(best_validation_ce_) +
            "\nACCURACY: " + str(best_accuracy_) +
            "\nCONFUSION MATRIX: " + str(best_conf_mx_))
    


if __name__ == "__main__":
    tf.app.run()
