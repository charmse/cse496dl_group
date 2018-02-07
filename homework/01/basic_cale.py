
import tensorflow as tf 
import numpy as np
import os

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', 'hackathon_3', 'directory where model graph and weights are saved')
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
    
    # specify the network
    x = tf.placeholder(tf.float32, [None, 784], name='data')
    with tf.name_scope('linear_model') as scope:
        hidden = tf.layers.dense(x,
                                 400,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 activation=tf.nn.relu,
                                 name='hidden_layer')
        output = tf.layers.dense(hidden,
                                 10,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                 name='output_layer')
        tf.identity(output, name='model_output')

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    red_mean = tf.reduce_mean(cross_entropy)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)


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
        min_validation_ce = 1.0
        count = 0
        for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))

            # run gradient steps and report mean loss on train data
            ce_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]       
                _, train_ce = session.run([train_op, red_mean], {x: batch_xs, y: batch_ys})
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))

            ce_vals = []
            for i in range(validation_num_examples // batch_size):
                batch_xs = validation_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = validation_labels[i*batch_size:(i+1)*batch_size, :]       
                validate_ce,_ = session.run([red_mean, y], {x: batch_xs, y: batch_ys})
                ce_vals.append(validate_ce)
            avg_validation_ce = sum(ce_vals) / len(ce_vals)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_validation_ce))

            # report mean test loss
            #ce_vals = []
            #conf_mxs = []
            #for i in range(test_num_examples // batch_size):
            #    batch_xs = test_images[i*batch_size:(i+1)*batch_size, :]
            #    batch_ys = test_labels[i*batch_size:(i+1)*batch_size, :]
            #    test_ce, conf_matrix = session.run([red_mean, confusion_matrix_op], {x: batch_xs, y: batch_ys})
            #    ce_vals.append(test_ce)
            #    conf_mxs.append(conf_matrix)
            #avg_test_ce = sum(ce_vals) / len(ce_vals)
            #print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
            #print('TEST CONFUSION MATRIX:')
            #print(str(sum(conf_mxs)))

            if avg_validation_ce < min_validation_ce:
                min_validation_ce = avg_validation_ce
                best_epoch = epoch
                best_train_ce = avg_train_ce
                #best_test_ce = avg_test_ce
                #best_conf_mxs = sum(conf_mxs)
                best_model = saver.save(session, os.path.join(FLAGS.save_dir, "mnist_inference"), global_step=global_step_tensor)
                count = 0
            else:
                count += 1
            
            if count > 12 :
                break

        print("EPOCH: " + str(best_epoch) +
              "\nTRAIN LOSS: " + str(best_train_ce) +
              "\nVALIDATION LOSS: " + str(min_validation_ce))
        #      "\nTEST LOSS: " + str(best_test_ce) +
        #      "\nCONFUSION MATRIX: " + str(best_conf_mxs))


if __name__ == "__main__":
    tf.app.run()
