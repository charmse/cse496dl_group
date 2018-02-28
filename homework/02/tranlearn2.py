import itertools as itr
import tensorflow as tf 
import numpy as np
import os
import sys
import util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse496dl/shared/homework/02/', 'directory where data is located')
flags.DEFINE_string('save_dir', '/work/pierolab/charper24/cse496dl_group/homework/02/models/', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 21, '')
flags.DEFINE_float('lr', 0.0001, '')
flags.DEFINE_integer('early_stop', 6, '')
flags.DEFINE_string('db', 'savee', '')
flags.DEFINE_integer('epoch_num', 100, '')
flags.DEFINE_float('reg_coeff', 0.001, '')
flags.DEFINE_float('split', 0.90, '')
FLAGS = flags.FLAGS

def make(graph):
    x = graph.get_tensor_by_name('input_placeholder:0')
    conv_out = graph.get_tensor_by_name('homework_02_model/conv_block_1/max_pooling2d_3/MaxPool:0')
    y = tf.placeholder(tf.float32, shape=[None, 7], name = 'label')
    flat_shape = int(np.prod(np.shape(conv_out)[1:]))
    conv_out1 = tf.reshape(tf.stop_gradient(conv_out),[-1, flat_shape])
    dense_1 = tf.layers.dense(conv_out1,1000,activation=tf.nn.relu , name ="dense_1")
    drop_1 = tf.layers.dropout(dense_1,0.8,name ="drop_1")
    dense_2 = tf.layers.dense(drop_1,500,activation=tf.nn.relu , name ="dense_2")
    drop_2 = tf.layers.dropout(dense_2,0.8,name ="drop_2")
    dense_3 = tf.layers.dense(drop_2,200,activation=tf.nn.relu , name ="dense_3")
    drop_3 = tf.layers.dropout(dense_3,0.8,name ="drop_3")
    output2 = tf.layers.dense(drop_3,7, name ="output2")
    return x,y,output2
    

def main(argv):

    # Set arguments:  Save_Dir Structure Learning_Rate Earling_Stoping Batch_Size Data_Dir    
    save_dir = FLAGS.save_dir
    learning_rate = FLAGS.lr
    early_stop = FLAGS.early_stop
    batch_size = FLAGS.batch_size
    reg_coeff = FLAGS.reg_coeff
    split = FLAGS.split
    if FLAGS.db == "savee":
        data_dir = FLAGS.data_dir + "SAVEE-British/"
        save_prefix = "savee_"
    else:
        data_dir = FLAGS.data_dir + "EMODB-German/"
        save_prefix = "emodb_"

    # # load training data
    train_images_1 = np.load(data_dir + 'train_x_1.npy')
    train_images_2 = np.load(data_dir + 'train_x_2.npy')
    train_images_3 = np.load(data_dir + 'train_x_3.npy')
    train_images_4 = np.load(data_dir + 'train_x_4.npy')
    train_labels_1 = np.load(data_dir + 'train_y_1.npy')
    train_labels_2 = np.load(data_dir + 'train_y_2.npy')
    train_labels_3 = np.load(data_dir + 'train_y_3.npy')
    train_labels_4 = np.load(data_dir + 'train_y_4.npy')

    # load testing data
    test_images_1 = np.load(data_dir + 'test_x_1.npy')
    test_images_2 = np.load(data_dir + 'test_x_2.npy')
    test_images_3 = np.load(data_dir + 'test_x_3.npy')
    test_images_4 = np.load(data_dir + 'test_x_4.npy')
    test_labels_1 = np.load(data_dir + 'test_y_1.npy')
    test_labels_2 = np.load(data_dir + 'test_y_2.npy')
    test_labels_3 = np.load(data_dir + 'test_y_3.npy')
    test_labels_4 = np.load(data_dir + 'test_y_4.npy')

    # split into train and validate
    train_images_1, valid_images_1, train_labels_1, valid_labels_1 = util.split_data(train_images_1, train_labels_1, split)
    train_images_2, valid_images_2, train_labels_2, valid_labels_2 = util.split_data(train_images_2, train_labels_2, split)
    train_images_3, valid_images_3, train_labels_3, valid_labels_3 = util.split_data(train_images_3, train_labels_3, split)
    train_images_4, valid_images_4, train_labels_4, valid_labels_4 = util.split_data(train_images_4, train_labels_4, split)

    #Create list of 
    train_images = [train_images_1, train_images_2, train_images_3, train_images_4]
    train_labels = [train_labels_1, train_labels_2, train_labels_3, train_labels_4]
    valid_images = [valid_images_1, valid_images_2, valid_images_3, valid_images_4]
    valid_labels = [valid_labels_1, valid_labels_2, valid_labels_3, valid_labels_4]
    test_images = [test_images_1, test_images_2, test_images_3, test_images_4]
    test_labels = [test_labels_1, test_labels_2, test_labels_3, test_labels_4]

    # specify the network

    # define classification loss
    
     #optimization

    # set up training and saving functionality
    
    #Open file to write to
    myfile = open('/work/pierolab/charper24/cse496dl_group/homework/02/' + 'output/' + save_prefix + 'model_'  + str(learning_rate) + '_' + str(batch_size) + '_' + str(early_stop) + '_out.txt', 'a+')

    #Create lists to collect best models
    best_epochs = []
    best_train_ces = []
    best_valid_ces = []
    best_valid_accuracies = []
    best_conf_mxs = []
    test_ces = []
    test_accuracies = []
    test_conf_mxs = []
    model_nos = []
    model_no = 1

    for train_images, train_labels, valid_images, valid_labels, test_images, test_labels in zip(train_images, train_labels, valid_images, valid_labels, test_images, test_labels):
        train_num_examples = train_images.shape[0]
        valid_num_examples = valid_images.shape[0]
        test_num_examples = test_images.shape[0]
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(save_dir + 'emodb_homework_2-0_C2:16,32,64;elu;l1;1.0;3;2|D:1000,500,200;relu;d;0.8_0.0001_52_12_4.meta')
            saver.restore(session,save_dir + 'emodb_homework_2-0_C2:16,32,64;elu;l1;1.0;3;2|D:1000,500,200;relu;d;0.8_0.0001_52_12_4')
            graph = session.graph
            x, y, output2 = make(graph) 

            with tf.name_scope('optimizer') as scope:
                cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output2)
                red_mean1 = tf.reduce_mean(cross_entropy1)
                Aoptimizer = tf.train.AdamOptimizer()
                train_op = Aoptimizer.minimize(cross_entropy1)
                confusion_matrix_op1 = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output2, axis=1), num_classes=7)
                accuracy_op1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output2, axis=1), tf.argmax(y, axis=1)) , tf.float32))   
            optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"optimizer")
            dense_vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"dense_1")
            drop_vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"drop_1")
            dense_vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"dense_2")
            drop_vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"drop_2")
            dense_vars_3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"dense_3")
            drop_vars_3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"drop_3")
            output_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"output2")
            session.run(tf.variables_initializer(optimizer_vars + dense_vars_1+drop_vars_1+dense_vars_2+drop_vars_2+dense_vars_3+drop_vars_3+output_vars, name = 'init'))


            # run training
            best_valid_ce = float("inf")
            count = 0
            for epoch in range(FLAGS.epoch_num):

                # run gradient steps and report mean loss on train data
                ce_vals = []
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]       
                    _, train_ce = session.run([train_op, red_mean1], {x: batch_xs, y: batch_ys})
                    ce_vals.append(train_ce)
                avg_train_ce = sum(ce_vals) / len(ce_vals)

                valid_accuracy_vals = []
                ce_vals = []
                for i in range(valid_num_examples // batch_size):
                    batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                    batch_ys = valid_labels[i*batch_size:(i+1)*batch_size, :]       
                    valid_ce, accuracy = session.run([red_mean1, accuracy_op1], {x: batch_xs, y: batch_ys})
                    ce_vals.append(valid_ce)
                    valid_accuracy_vals.append(accuracy)
                avg_valid_ce = sum(ce_vals) / len(ce_vals)
                avg_valid_accuracy = sum(valid_accuracy_vals) / len(valid_accuracy_vals)

                myfile.write("Epoch: " + str(epoch) +
                "\nTrain loss: " + str(avg_train_ce) +
                "\nValidation loss: " + str(avg_valid_ce) +
                "\nAccuracy: " + str(avg_valid_accuracy) +
                "\n------------------------------\n")

                if avg_valid_ce < best_valid_ce:
                    best_valid_ce = avg_valid_ce
                    best_epoch = epoch
                    best_train_ce = avg_train_ce
                    best_valid_accuracy = avg_valid_accuracy
                    best_model = saver.save(session, os.path.join(save_dir + "models/", save_prefix + "homework_2-0_" + str(learning_rate) + '_' + str(batch_size) + '_' + str(early_stop) +  "_" + str(model_no)))
                    count = 0
                else:
                    count += 1

                if count > early_stop:
                    break
            
            myfile.write("BEST VALIDATION CROSS-ENTROPY" +
            "\n-----------------------------" +
            "\nEPOCH: " + str(best_epoch) +
            "\nTRAIN LOSS: " + str(best_train_ce) +
            "\nVALIDATION LOSS: " + str(best_valid_ce) +
            "\nACCURACY: " + str(best_valid_accuracy) +
            "\n------------------------------------------\n")

            ce_vals = []
            conf_mxs = []
            test_accuracy_vals = []
            for i in range(test_num_examples // batch_size):
                batch_xs = test_images[i*batch_size:(i+1)*batch_size, :]
                batch_ys = test_labels[i*batch_size:(i+1)*batch_size, :]       
                test_ce, conf_matrix, test_accuracy = session.run([red_mean1, confusion_matrix_op1, accuracy_op1], {x: batch_xs, y: batch_ys})
                ce_vals.append(test_ce)
                conf_mxs.append(conf_matrix)
                test_accuracy_vals.append(test_accuracy)
            avg_test_ce = sum(ce_vals) / len(ce_vals)
            avg_test_accuracy = sum(test_accuracy_vals) / len(test_accuracy_vals)

            myfile.write("TEST RESULTS" +
            "\n-----------------------------" +
            "\nTEST LOSS: " + str(avg_test_ce) +
            "\nACCURACY: " + str(avg_test_accuracy) +
            "\nCONFUSION MATRIX: \n" + str(sum(conf_mxs)) +
            "\n------------------------------------------\n")


            
        #Collect best's
        best_epochs.append(best_epoch)
        best_train_ces.append(best_train_ce)
        best_valid_ces.append(best_valid_ce)
        best_valid_accuracies.append(best_valid_accuracy)
        test_ces.append(avg_test_ce)
        test_accuracies.append(avg_test_accuracy)
        test_conf_mxs.append(sum(conf_mxs))
        model_nos.append(model_no)
        model_no += 1

        tf.reset_default_graph()

    for model_no, epoch, train_ce, valid_accuracy, validate_ce, test_accuracy, test_ce in zip(model_nos, best_epochs, best_train_ces, best_valid_accuracies, best_valid_ces, test_accuracies, test_ces):
        allfile.write("{" + arch + '},' + str(model_no) + ',' + str(learning_rate) + ',' + str(early_stop) + ',' + str(batch_size) + ',' + str(epoch) + ',' + str(train_ce) + ',' + str(valid_accuracy) + ',' + str(validate_ce) + ',' + str(test_accuracy) + ',' + str(test_ce) +"\n")

    myfile.write("AVERAGE AND STANDARD DEVIATION OF CROSS VALIDATION" +
    "\n-----------------------------" +
    "\nEPOCH: " + str(np.average(best_epochs)) + "  ,  " + str(np.std(best_epochs)) +
    "\nTRAIN LOSS: " + str(np.average(best_train_ces)) + "  ,  " + str(np.std(best_train_ces)) +
    "\nVALIDATION LOSS: " + str(np.average(best_valid_ces)) + "  ,  " + str(np.std(best_valid_ces)) +
    "\nVALIDATION ACCURACY: " + str(np.average(best_valid_accuracies)) + "  ,  " + str(np.std(best_valid_accuracies)) +
    "\nTEST LOSS: " + str(np.average(test_ces)) + "  ,  " + str(np.std(test_ces)) +
    "\nTEST ACCURACY: " + str(np.average(test_accuracies)) + "  ,  " + str(np.std(test_accuracies)) +
    "\nCONFUSION MATRIX: \n" + str(sum(test_conf_mxs)) +
    "\n------------------------------------------\n")

    myfile.close()
    allfile.close()

if __name__ == "__main__":
    tf.app.run()
