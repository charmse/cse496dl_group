import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import KFold
#import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where fashion-MNIST is located')
flags.DEFINE_string('save_dir', 'homework01', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_epoch_num', 50, '')
FLAGS = flags.FLAGS
#Split the training data in to train_data and validation_data
def split_data(data, proportion):
    """
    Split a numpy array into two parts of `proportion` and `1 - proportion`
    Args:
    - data: numpy array, to be split along the first axis
    - proportion: a float less than 1
    """
    size = data.shape[0]
    np.random.seed(42)
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[s[:split_idx]], data[s[split_idx:]]
def onehotcode(data):
    data_final = np.zeros((data.shape[0],10))
    data_final[np.arange(data.shape[0]),data.astype(int)]=1
    return data_final
def main(argv):
    # load data
    images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
    labels = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')
    labels = onehotcode(labels)
    #train_data,validation_data = split_data(images,0.9)
    #train_labels,validation_labels = split_data(labels,0.9)
    #train_data_num = train_data.shape[0]
    #validation_data_num = validation_data.shape[0]
    kf = KFold(n_splits  =10,shuffle = False,random_state=None)
    #kf.get_n_splits(images)
    #print(train_data_num)
    #print(train_labels.shape)
    # specify the network
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    with tf.name_scope('linear_model') as scope:
        hidden = tf.layers.dense(x, 500,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.5),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.5),
                                         activation=tf.nn.relu, name='hidden_layer1')
        hidden2 = tf.layers.dense(hidden, 300,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.5),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.5),
                                         activation=tf.nn.relu, name='hidden_layer2')
        hidden3 = tf.layers.dense(hidden2, 200,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.5),
                                         bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.5),
                                         activation=tf.nn.relu, name='hidden_layer3')
        output = tf.layers.dense(hidden3, 10,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.5),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=1.), name='output_layer')
        tf.identity(output, name='output')

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='actual_labels')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y,axis=1), tf.argmax(output, axis=1), num_classes=10)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # this is the weight of the regularization part of the final loss
    REG_COEFF = 0.001
    total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()
    c_entropy = tf.reduce_mean(total_loss)
    for train_index, val_index in kf.split(images):
        train_data,validation_data = images[train_index],images[val_index]
        train_labels,validation_labels = labels[train_index],labels[val_index]
        train_data_num = train_data.shape[0]
        validation_data_num = validation_data.shape[0]
        print('New Fold: -----------------------------------')
        with tf.Session() as session:
             session.run(tf.global_variables_initializer())
             #session.run(tf.initialize_all_variables())
        #prev_validation_ce = float("inf")
             best_validation_ce = float("inf")
        # run training
             batch_size = FLAGS.batch_size
             test_loss = []
             for epoch in range(FLAGS.max_epoch_num):
                 print('Epoch: ' + str(epoch))
                 # run gradient steps and report mean loss on train data
                 ce_vals = []
                 for i in range(train_data_num // batch_size):
                     batch_xs = train_data[i*batch_size:(i+1)*batch_size, :]
                     batch_ys = train_labels[i*batch_size:(i+1)*batch_size,:]
                     _, train_ce = session.run([train_op, c_entropy], {x: batch_xs, y: batch_ys})
                     ce_vals.append(train_ce)
                 avg_train_ce = sum(ce_vals) / len(ce_vals)
                 print('TRAIN LOSS VALUE: ' + str(avg_train_ce))

            #run the validation tests
            #if(epoch >0):
            #    prev_validation_ce = avg_validation_ce
                 ce_vals = []
                 conf_mxs = []
                 for i in range(validation_data_num // batch_size):
                     batch_xs = validation_data[i*batch_size:(i+1)*batch_size, :]
                     batch_ys = validation_labels[i*batch_size:(i+1)*batch_size, :]
                     val_test_ce,conf_matrix = session.run([c_entropy,confusion_matrix_op], {x: batch_xs, y: batch_ys})
                     ce_vals.append(val_test_ce)
                     conf_mxs.append(conf_matrix)
                 avg_validation_ce = sum(ce_vals) / len(ce_vals)
                 print('VALIDATION TOTAL LOSS: ' + str(avg_validation_ce))
            #print('validation confusion matrix')
            #print(str(sum(conf_mxs)))
            #if(epoch ==0):
            #   prev_validation_ce = avg_validation_ce
            #print('previous validation loss:   ' + str(prev_validation_ce))
                 if(best_validation_ce > avg_validation_ce):
                     best_epoch = epoch
                     counter =0
                     best_validation_ce = avg_validation_ce
                     path_prefix = saver.save(session,os.path.join(FLAGS.save_dir,"homework_1"), global_step=global_step_tensor)
                 else:
                     counter +=1
                 print('Best Validation error:'+str(best_validation_ce))
                 print(counter)
            #ce_vals = []
            #conf_mxs = []
            #for i in range(test_data_num // batch_size):
            #    batch_xs = test_data[i*batch_size:(i+1)*batch_size, :]
            #    batch_ys = test_labels[i*batch_size:(i+1)*batch_size,:]
            #    test_ce, conf_matrix = session.run([c_entropy, confusion_matrix_op], {x: batch_xs, y: batch_ys})
            #    ce_vals.append(test_ce)
            #    conf_mxs.append(conf_matrix)
            #avg_test_ce = sum(ce_vals) / len(ce_vals)
            #print('TEST LOSS VALUE: ' + str(avg_test_ce))
            #test_loss = np.append(test_loss,avg_test_ce)
                 if(counter>8):
                     break
        #a = np.arange(0.,FLAGS.max_epoch_num,1.)
        #plt.plot(a,test_loss,'r--')
        #plt.show()
        #path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, "fmnist_inference"), global_step=global_step_tensor)

if __name__ == "__main__":
    tf.app.run()

