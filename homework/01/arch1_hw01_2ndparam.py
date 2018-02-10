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
    FOLDS = 10
    NEURONS_HIDDEN1 = 300
    NEURONS_HIDDEN2 = 200
    NEURONS_HIDDEN3 = 200
    NEURONS_HIDDEN4 = 100
    KEEP_PROB = 0.5
    REG_SCALE = 0.6
    kf = KFold(n_splits  =FOLDS,shuffle = False,random_state=None)
    # specify the network
    x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
    with tf.name_scope('linear_model') as scope:
        #dropped_input = tf.layers.dropout(x, KEEP_PROB)
        hidden1 = tf.layers.dense(x, NEURONS_HIDDEN1,activation=tf.nn.relu, name='hidden_layer1')
        #dropped_hidden1 = tf.layers.dropout(hidden1, KEEP_PROB)
        hidden2 = tf.layers.dense(hidden1, NEURONS_HIDDEN2,activation = tf.nn.relu,name = 'hidden_layer2')
        #dropped_hidden2 = tf.layers.dropout(hidden2, KEEP_PROB)
        #hidden3 = tf.layers.dense(dropped_hidden2, NEURONS_HIDDEN3,activation=tf.nn.relu, name='hidden_layer3')
        #dropped_hidden3 = tf.layers.dropout(hidden3, KEEP_PROB)
        #hidden4 = tf.layers.dense(dropped_hidden3, NEURONS_HIDDEN4,activation=tf.nn.relu, name='hidden_layer4')
        #dropped_hidden4 = tf.layers.dropout(hidden4,KEEP_PROB)
        output = tf.layers.dense(hidden2, 10, name='output_layer')
    tf.identity(output, name='output')

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 10], name='actual_labels')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y,axis=1), tf.argmax(output, axis=1), num_classes=10)
    #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # this is the weight of the regularization part of the final loss
    #REG_COEFF = 0.001
    output_softmax = tf.nn.softmax(output)
    prediction = tf.argmax(output_softmax, 1)
    label = tf.argmax(y,1)
    equality = tf.equal(prediction, label)
    accuracy_op = tf.reduce_mean(tf.cast(equality, tf.float32))
    total_loss = cross_entropy

    # set up training and saving functiona lity
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()
    c_entropy = tf.reduce_mean(total_loss)
    all_best = []
    for i in [400]:
        print('**********************************************************************')
        for train_index, val_index in kf.split(images):
            train_data,validation_data = images[train_index],images[val_index]
            train_labels,validation_labels = labels[train_index],labels[val_index]
            train_data_num = train_data.shape[0]
            validation_data_num = validation_data.shape[0]
            train_data = train_data/255
            validation_data = validation_data/255
            print(train_data_num)
            print(validation_data_num)
            print('New Fold: -----------------------------------')
            with tf.Session() as session:
                 session.run(tf.global_variables_initializer())
             #session.run(tf.initialize_all_variables())
                 best_validation_ce = float("inf")
                 batch_size = FLAGS.batch_size
                 test_loss = []
                 accuracy_vals =[]
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
                     ce_vals = []
                     conf_mxs = []
                     for i in range(validation_data_num // batch_size):
                         batch_xs = validation_data[i*batch_size:(i+1)*batch_size, :]
                         batch_ys = validation_labels[i*batch_size:(i+1)*batch_size, :]
                         accuracy = session.run(accuracy_op, {x: batch_xs, y: batch_ys})
                         val_test_ce,conf_matrix = session.run([c_entropy,confusion_matrix_op], {x: batch_xs, y: batch_ys})
                         ce_vals.append(val_test_ce)
                         conf_mxs.append(conf_matrix)
                         accuracy_vals.append(accuracy)
                     avg_validation_ce = sum(ce_vals) / len(ce_vals)
                     final_accuracy = np.sum(accuracy_vals)/len(accuracy_vals)
                     print('VALIDATION TOTAL LOSS: ' + str(avg_validation_ce))
            #print('validation confusion matrix')
            #print(str(sum(conf_mxs)))
                     if(best_validation_ce > avg_validation_ce):
                         best_epoch = epoch
                         counter =0
                         best_validation_ce = avg_validation_ce
                         best_accuracy = final_accuracy
                         path_prefix = saver.save(session,os.path.join(FLAGS.save_dir,"homework_1"), global_step=global_step_tensor)
                     else:
                         counter +=1
                     print('Best Validation error:'+str(best_validation_ce))
                     print(counter)
                     print('BEST ACCURACY:'+str(best_accuracy))
                     if(counter>12):
                         all_best.append(best_validation_ce)
                         break
        print(NEURONS_HIDDEN1)
        avg_best = sum(all_best) / len(all_best)
        print('AVERAGE OF ALL K-FLODS :  '+str(avg_best))
        #a = np.arange(0.,FLAGS.max_epoch_num,1.)
        #plt.plot(a,test_loss,'r--')
        #plt.show()
        #path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, "fmnist_inference"), global_step=global_step_tensor)

if __name__ == "__main__":
    tf.app.run()

