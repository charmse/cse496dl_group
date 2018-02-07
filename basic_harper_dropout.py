File Edit Options Buffers Tools Python Help                                                                                                                                                                                                                                                           
import tensorflow as tf
import numpy as np
import os

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/01/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '/work/pierolab/charper24/csce496/hw/hw01', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('max_epoch_num', 100, '')
FLAGS = flags.FLAGS

def split_data(data, labels, proportion):
    """                                                                                                                                                                                                                                                                                               
    Split a numpy array into two parts of `proportion` and `1 - proportion`                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                      
    Args:                                                                                                                                                                                                                                                                                             
        - data: numpy array, to be split along the first axis                                                                                                                                                                                                                                         
        - labels: numpy array to be split in the same way that data is split                                                                                                                                                                                                                          
        - proportion: a float less than 1                                                                                                                                                                                                                                                             
    """
    size = data.shape[0]
    np.random.seed(42)
    indicies = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[indicies[:split_idx]], data[indicies[split_idx:]], labels[indicies[:split_idx]], labels[indicies[split_idx:]]

def one_hot_label(labels):
    """                                                                                                                                                                                                                                                                                               
    Convert the integer label values to a one-hot encoding array                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                      
    Args:                                                                                                                                                                                                                                                                                             
        - data: numpy array of integers to be converted to one-hot                                                                                                                                                                                                                                    
    """
    label_final = np.zeros((labels.shape[0],10))
    label_final[np.arange(labels.shape[0]),labels.astype(int)]=1
    return label_final

def main(argv):
    # load data                                                                                                                                                                                                                                                                                       
    train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
    train_labels_before = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')
    train_labels = one_hot_label(train_labels_before)

    # split into train and validate                                                                                                                                                                                                                                                                   
    train_img, valid_img, train_lbl, valid_lbl = split_data(train_images, train_labels, .90)

    train_num_examples = train_img.shape[0]
    valid_num_examples = valid_img.shape[0]

    NEURONS_HIDDEN = 400

    # specify the network                                                                                                                                                                                                                                                                             
    x = tf.placeholder(tf.float32, [None, 784], name='data')
    x_norm = x/255
    KEEP_PROB = 0.8

    with tf.name_scope('linear_model') as scope:
        dropped_input = tf.layers.dropout(x_norm, KEEP_PROB)
        hidden1 = tf.layers.dense(dropped_input,
                                 NEURONS_HIDDEN,
                                 activation=tf.nn.relu,
                                 name='hidden_layer1')
        dropped_hidden1 = tf.layers.dropout(hidden1, KEEP_PROB)
        hidden2 = tf.layers.dense(dropped_input,
                                  NEURONS_HIDDEN,
                                  activation=tf.nn.relu,
                                  name='hidden_layer2')
        dropped_hidden2 = tf.layers.dropout(hidden2, KEEP_PROB)
        output = tf.layers.dense(dropped_hidden2,
                                 10,
                                 name='output_layer')
        tf.identity(output, name='model_output')

    # define classification loss                                                                                                                                                                                                                                                                      
    y = tf.placeholder(tf.float32, [None, 10], name='label')
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

    # set up training and saving functionality                                                                                                                                                                                                                                                        
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # run training                                                                                                                                                                                                                                                                                
        batch_size = FLAGS.batch_size
        best_epoch = 0
        best_valid_ce = float("inf")
        patience_param = 10
        ephocs_wo_improve = 0

        #Define new tensor                                                                                                                                                                                                                                                                            
        cross_entr_mean = tf.reduce_mean(cross_entropy)

        for epoch in range(FLAGS.max_epoch_num):
            print('Epoch: ' + str(epoch))

            # run gradient steps and report mean loss on train data                                                                                                                                                                                                                                   
            ce_vals = []
            for i in range(train_num_examples // batch_size):
                batch_xs = train_img[i*batch_size:(i+1)*batch_size, :]
                batch_ys = train_lbl[i*batch_size:(i+1)*batch_size, :]
                _, train_ce = session.run([train_op, cross_entr_mean], {x: batch_xs, y: batch_ys})
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)
            print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))

            # run gradient steps and report mean loss on validation data                                                                                                                                                                                                                              
            ce_vals = []
            conf_mxs = []
            for i in range(valid_num_examples // batch_size):
                batch_xs = valid_img[i*batch_size:(i+1)*batch_size, :]
                batch_ys = valid_lbl[i*batch_size:(i+1)*batch_size, :]
                valid_ce, conf_matrix = session.run([cross_entr_mean, confusion_matrix_op], {x: batch_xs, y: batch_ys})
                ce_vals.append(valid_ce)
                conf_mxs.append(conf_matrix)
            avg_valid_ce = sum(ce_vals) / len(ce_vals)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_valid_ce))
            print(str(sum(conf_mxs)))

            #Save best and Early stopping                                                                                                                                                                                                                                                             
            if(best_valid_ce > avg_valid_ce):
                best_valid_ce = avg_valid_ce
                best_train_ce = avg_train_ce
                best_epoch = epoch
                path_prefix = saver.save(session, os.path.join(FLAGS.save_dir, "homework_1"), global_step=global_step_tensor)
                epochs_wo_improve = 0
            else:
                epochs_wo_improve += 1
            #Check for early stopping                                                                                                                                                                                                                                                                 
            if(epochs_wo_improve > patience_param):
                break

        print('BEST EPOCH: ' + str(best_epoch))
        print('BEST TRAIN CROSS ENTROPY: ' + str(best_train_ce))
        print('BEST VALIDATION CROSS ENTROPY: ' + str(best_valid_ce))

        #Append model parameters and results to a text file                                                                                                                                                                                                                                           
        with open('/work/pierolab/charper24/csce496/hw/hw01/modelParamsAndOut.txt', "a") as myfile:
            myfile.write('-------------------------------------------- \n')
            #myfile.write('REG COEFF: ' + str(REG_COEFF) + ' \n')                                                                                                                                                                                                                                     
            #myfile.write('REG SCALE: ' + str(REG_SCALE) + ' \n')                                                                                                                                                                                                                                     
            myfile.write('BATCH SIZE: ' + str(batch_size) + '\n')
            myfile.write('KEEP PROB: ' + str(KEEP_PROB) + '\n')
            myfile.write('# Hidden Layers: 2 \n')
            myfile.write('Neurons in Hidden: ' + str(NEURONS_HIDDEN) + '\n')
            myfile.write('Regularization Type: Dropout \n')
            myfile.write('     BEST EPOCH: ' + str(best_epoch) + '\n')
            myfile.write('     BEST TRAIN CROSS ENTROPY: ' + str(best_train_ce) + '\n')
            myfile.write('     BEST VALID CROSS ENTROPY: ' + str(best_valid_ce) + '\n')


if __name__ == "__main__":
    tf.app.run()
#Try implementing some data augmentation                                                                                                                                                                                                                                                              






