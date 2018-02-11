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
    size = data.shape[0]
    np.random.seed(42)
    indicies = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[indicies[:split_idx]], data[indicies[split_idx:]], labels[indicies[:split_idx]], labels[indicies[split_idx:]]

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
  
    train_images, validation_images, train_labels, validation_labels = split_data(train_images, train_labels, .90)

    validation_num_examples = validation_images.shape[0]
    train_num_examples = train_images.shape[0]
    
    """
        Model using three layers of 500 neurons
        with 0.001 learning rate
        and dropout using 0.8 as Keep probability
    """
    # specify the network
    input_placeholder = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    input_norm = input_placeholder/255
    KEEP_PROB = 0.8

    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(input_norm, KEEP_PROB)
        hidden_1 = tf.layers.dense(dropped_input,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_1')
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        hidden_2 = tf.layers.dense(dropped_hidden_1,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_2')
        dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
        hidden_3 = tf.layers.dense(dropped_hidden_2,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_3')
        dropped_hidden_3 = tf.layers.dropout(hidden_3, KEEP_PROB)
        output = tf.layers.dense(dropped_hidden_3,
                                    10,
                                    name='output_layer')
    tf.identity(output, name='output')

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    red_mean = tf.reduce_mean(cross_entropy)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)) , tf.float32))


    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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

            with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
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
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_8"))
                count = 0
            else:
                count += 1

            if count > 10:
                break

        with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
            myfile.write(
            "BEST VALIDATION CROSS-ENTROPY" +
            "\n-------------------------------------------------"
            "\nThree 500 | Learning Rate : 0.001 | Dropout : 0.8"
            "\n-------------------------------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")
    
    tf.reset_default_graph()

    """
        Model using three layers of 500 neurons
        with 0.0005 learning rate
        and dropout using 0.8 as Keep probability
    """
    # specify the network
    input_placeholder = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    input_norm = input_placeholder/255
    KEEP_PROB = 0.8

    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(input_norm, KEEP_PROB)
        hidden_1 = tf.layers.dense(dropped_input,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_1')
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        hidden_2 = tf.layers.dense(dropped_hidden_1,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_2')
        dropped_hidden_2 = tf.layers.dropout(hidden_2, KEEP_PROB)
        hidden_3 = tf.layers.dense(dropped_hidden_2,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_3')
        dropped_hidden_3 = tf.layers.dropout(hidden_3, KEEP_PROB)
        output = tf.layers.dense(dropped_hidden_3,
                                    10,
                                    name='output_layer')
    tf.identity(output, name='output')

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

            with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
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
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_8"))
                count = 0
            else:
                count += 1

            if count > 10:
                break

        with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
            myfile.write(
            "BEST VALIDATION CROSS-ENTROPY" +
            "\n--------------------------------------------------"
            "\nThree 500 | Learning Rate : 0.0005 | Dropout : 0.8"
            "\n--------------------------------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")
    
    tf.reset_default_graph()

    """
        Model using three layers of 500 neurons
        with 0.001 learning rate
        and not using dropout
    """
    # specify the network
    input_placeholder = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    input_norm = input_placeholder/255

    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(input_norm,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_1')
        hidden_2 = tf.layers.dense(hidden_1,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_2')
        hidden_3 = tf.layers.dense(hidden_2,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_3')
        output = tf.layers.dense(hidden_3,
                                    10,
                                    name='output_layer')
    tf.identity(output, name='output')

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    red_mean = tf.reduce_mean(cross_entropy)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)) , tf.float32))


    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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

            with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
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
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_8"))
                count = 0
            else:
                count += 1

            if count > 10:
                break

        with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
            myfile.write(
            "BEST VALIDATION CROSS-ENTROPY" +
            "\n-------------------------------------------------"
            "\nThree 500 | Learning Rate : 0.001 | Dropout : N/A"
            "\n-------------------------------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")
    
    tf.reset_default_graph()

    """
        Model using three layers of 500 neurons
        with 0.0005 learning rate
        and not using dropout
    """
    # specify the network
    input_placeholder = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    input_norm = input_placeholder/255

    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(input_norm,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_1')
        hidden_2 = tf.layers.dense(hidden_1,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_2')
        hidden_3 = tf.layers.dense(hidden_2,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_3')
        output = tf.layers.dense(hidden_3,
                                    10,
                                    name='output_layer')
    tf.identity(output, name='output')

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

            with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
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
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_8"))
                count = 0
            else:
                count += 1

            if count > 10:
                break

        with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
            myfile.write(
            "BEST VALIDATION CROSS-ENTROPY" +
            "\n--------------------------------------------------"
            "\nThree 500 | Learning Rate : 0.0005 | Dropout : N/A"
            "\n--------------------------------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")
    
    tf.reset_default_graph()

    """
        Model using one layer of 500 neurons
        with 0.001 learning rate
        and dropout using 0.8 as Keep probability
    """
    # specify the network
    input_placeholder = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    input_norm = input_placeholder/255
    KEEP_PROB = 0.8

    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(input_norm, KEEP_PROB)
        hidden_1 = tf.layers.dense(dropped_input,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_1')
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        output = tf.layers.dense(dropped_hidden_1,
                                    10,
                                    name='output_layer')
    tf.identity(output, name='output')

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    red_mean = tf.reduce_mean(cross_entropy)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)) , tf.float32))


    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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

            with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
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
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_8"))
                count = 0
            else:
                count += 1

            if count > 10:
                break

        with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
            myfile.write(
            "BEST VALIDATION CROSS-ENTROPY" +
            "\n-------------------------------------------------"
            "\nOne 500 | Learning Rate : 0.001 | Dropout : 0.8"
            "\n-------------------------------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")
    
    tf.reset_default_graph()

    """
        Model using one layer of 500 neurons
        with 0.0005 learning rate
        and dropout using 0.8 as Keep probability
    """
    # specify the network
    input_placeholder = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    input_norm = input_placeholder/255
    KEEP_PROB = 0.8

    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(input_norm, KEEP_PROB)
        hidden_1 = tf.layers.dense(dropped_input,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_1')
        dropped_hidden_1 = tf.layers.dropout(hidden_1, KEEP_PROB)
        output = tf.layers.dense(dropped_hidden_1,
                                    10,
                                    name='output_layer')
    tf.identity(output, name='output')

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

            with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
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
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_8"))
                count = 0
            else:
                count += 1

            if count > 10:
                break

        with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
            myfile.write(
            "BEST VALIDATION CROSS-ENTROPY" +
            "\n--------------------------------------------------"
            "\nOne 500 | Learning Rate : 0.0005 | Dropout : 0.8"
            "\n--------------------------------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")
    
    tf.reset_default_graph()

    """
        Model using one layer of 500 neurons
        with 0.001 learning rate
        and not using dropout
    """
    # specify the network
    input_placeholder = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    input_norm = input_placeholder/255

    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(input_norm,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_1')
        output = tf.layers.dense(hidden_1,
                                    10,
                                    name='output_layer')
    tf.identity(output, name='output')

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    red_mean = tf.reduce_mean(cross_entropy)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)) , tf.float32))


    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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

            with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
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
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_8"))
                count = 0
            else:
                count += 1

            if count > 10:
                break

        with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
            myfile.write(
            "BEST VALIDATION CROSS-ENTROPY" +
            "\n-------------------------------------------------"
            "\nOne 500 | Learning Rate : 0.001 | Dropout : N/A"
            "\n-------------------------------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")
    
    tf.reset_default_graph()

    """
        Model using one layer of 500 neurons
        with 0.0005 learning rate
        and not using dropout
    """
    # specify the network
    input_placeholder = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    input_norm = input_placeholder/255

    with tf.name_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(input_norm,
                                    500,
                                    activation=tf.nn.relu,
                                    name='hidden_layer_1')
        output = tf.layers.dense(hidden_1,
                                    10,
                                    name='output_layer')
    tf.identity(output, name='output')

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

            with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
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
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1-0_8"))
                count = 0
            else:
                count += 1

            if count > 10:
                break

        with open('/work/soh/charms/cse496dl/homework/01/homework_8.txt', 'a') as myfile:
            myfile.write(
            "BEST VALIDATION CROSS-ENTROPY" +
            "\n--------------------------------------------------"
            "\nOne 500 | Learning Rate : 0.0005 | Dropout : N/A"
            "\n--------------------------------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_validation_ce) +
            "\nACCURACY: " + str(best_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(best_conf_mx) +
            "\n------------------------------------------\n")
    

if __name__ == "__main__":
    tf.app.run()
