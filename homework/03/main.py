import itertools as itr
import tensorflow as tf 
import numpy as np
import os
import sys
import model
import util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_string('arch', 'C2:16,32,64;elu;l1;1.0;3;2|D:1000,500,250;elu;d;0.8', '')
flags.DEFINE_integer('early_stop', 12, '')
flags.DEFINE_string('db', 'emodb', '')
flags.DEFINE_integer('epoch_num', 100, '')
flags.DEFINE_float('reg_coeff', 0.001, '')
flags.DEFINE_float('split', 0.90, '')
flags.DEFINE_string('transfer', '', '')
flags.DEFINE_string('ae', '', '')
flags.DEFINE_integer('code_size', 100, '')
flags.DEFINE_float('sparsity_weight', 5e-3, '')
flags.DEFINE_boolean('validate', False, '')
FLAGS = flags.FLAGS

def main(argv):

    # Set arguments:  Save_Dir Structure Learning_Rate Earling_Stoping Batch_Size Data_Dir    
    arch = FLAGS.arch
    save_dir = FLAGS.save_dir
    learning_rate = FLAGS.lr
    early_stop = FLAGS.early_stop
    batch_size = FLAGS.batch_size
    reg_coeff = FLAGS.reg_coeff
    split = FLAGS.split
    transfer = FLAGS.transfer
    ae = FLAGS.ae
    code_size = FLAGS.code_size
    sparsity_weight = FLAGS.sparsity_weight
    validate = FLAGS.validate
    if FLAGS.db == "savee":
        data_dir = FLAGS.data_dir + "SAVEE-British/"
        save_prefix = "savee_"
    elif FLAGS.db == 'emodb':
        data_dir = FLAGS.data_dir + "EMODB-German/"
        save_prefix = "emodb_"
    else:
        data_dir = FLAGS.data_dir + "cifar-100/"
        save_prefix = "cifar-100_"

    # specify the network
    if bool(transfer):
        x, output, arch = model.transfer(save_dir + 'models/' + transfer)
        opt_name = 'new_optimizer'
    elif bool(ae):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='encoder_input')
        code, outputs, ae_name = model.autoencoder_network(x, code_size=code_size, model=ae)
        arch = 'AE'
    else:
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
        output = model.make(x,arch)
        opt_name = 'optimizer'
    

    # # load training data
    train_images = np.load(data_dir + 'x_train.npy')
    train_labels = np.load(data_dir + 'y_train.npy')
    # load testing data
    test_images = np.load(data_dir + 'x_test.npy')
    test_labels = np.load(data_dir + 'y_test.npy')

    # split into train and validate
    if validate:
        train_images, valid_images, train_labels, valid_labels = util.split_data(train_images, train_labels, split)
        valid_num_examples = valid_images.shape[0]
    train_num_examples = train_images.shape[0]
    test_num_examples = test_images.shape[0]

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 100], name='label')

    if bool(ae):

        # calculate loss
        sparsity_loss = tf.norm(code, ord=1, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.square(outputs - x)) # MSE
        total_loss = reconstruction_loss + sparsity_weight * sparsity_loss

        # setup optimizer
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(total_loss)

        encoder_saver = tf.train.Saver()
        decoder_saver = tf.train.Saver()

        with tf.Session() as session:
            
            #initialize variables
            session.run(tf.global_variables_initializer())

            best_valid_loss = float("inf")
            count = 0
            for epoch in range(FLAGS.epoch_num):
                
                #Train
                for i in range(train_num_examples // batch_size):
                    batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                    session.run(train_op, {x: batch_xs})
                
                #Validate
                if validate:
                    valid_loss = []
                    for i in range(valid_num_examples // batch_size):
                        batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                        loss = session.run(total_loss, {x: batch_xs})
                        valid_loss.append(loss)
                    avg_valid_loss = np.sum(valid_loss) / len(valid_loss)
                    
                    #Early Stopping
                    if avg_valid_loss < best_valid_loss:
                        best_valid_loss = avg_valid_loss
                        best_epoch = epoch
                        count = 0
                        encoder_saver.save(session, os.path.join(save_dir + "models/", "maxquality_encoder_homework_3-0"))
                        decoder_saver.save(session, os.path.join(save_dir + "models/", "maxquality_decoder_homework_3-0"))
                        encoder_saver.save(session, os.path.join(save_dir + "models/", "maxcompression_encoder_homework_3-0"))
                        decoder_saver.save(session, os.path.join(save_dir + "models/", "maxcompression_decoder_homework_3-0"))
                    else:
                        count += 1

                    if count > early_stop:
                        break

            #Run a test
            psnr_list = []
            for i in range(test_num_examples): 
                x_out, code_out, output_out = session.run([x, code, outputs], {x: np.expand_dims(train_images[i], axis=0)})
                psnr_list.append(util.psnr(train_images[i],output_out))
            avg_psnr = np.sum(psnr_list) / len(psnr_list)

            if not validate:
                encoder_saver.save(session, os.path.join(save_dir + "models/", "maxquality_encoder_homework_3-0"))
                decoder_saver.save(session, os.path.join(save_dir + "models/", "maxquality_decoder_homework_3-0"))
                encoder_saver.save(session, os.path.join(save_dir + "models/", "maxcompression_encoder_homework_3-0"))
                decoder_saver.save(session, os.path.join(save_dir + "models/", "maxcompression_decoder_homework_3-0"))

        allfile = open('output/all_models_out.csv', 'a+')
        allfile.write(ae_name + ',' +str(code_size) + ',' + str(sparsity_weight) + ',' + str(batch_size) + ',' + str(epoch) + ',' + str(avg_psnr) +"\n")
        allfile.close()

    else:

        with tf.name_scope(opt_name) as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
            red_mean = tf.reduce_mean(cross_entropy)
            #global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)    
            if bool(transfer):
                total_loss = cross_entropy
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name = 'transfer_optimizer')
            else:
                regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                total_loss = cross_entropy + reg_coeff * sum(regularization_losses)
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            train_op = optimizer.minimize(total_loss)
            confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=7)
            accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1)) , tf.float32))
        

        saver = tf.train.Saver()

        #Open file to write to
        myfile = open(save_dir + 'output/' + save_prefix + 'model_' + arch + '_' + str(learning_rate) + '_' + str(batch_size) + '_' + str(early_stop) + '_out.txt', 'a+')
        allfile = open('output/all_models_out.csv', 'a+')

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

                if bool(transfer):
                    optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"new_optimizer")
                    new_dense_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"dense_block_new")
                    output_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"output2")
                    session.run(tf.variables_initializer(optimizer_vars + new_dense_vars + output_vars))
                else:
                    session.run(tf.global_variables_initializer())

                # run training
                best_valid_ce = float("inf")
                count = 0
                for epoch in range(FLAGS.epoch_num):

                    # run gradient steps and report mean loss on train data
                    ce_vals = []
                    for i in range(train_num_examples // batch_size):
                        batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
                        batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]       
                        _, train_ce = session.run([train_op, red_mean], {x: batch_xs, y: batch_ys})
                        ce_vals.append(train_ce)
                    avg_train_ce = sum(ce_vals) / len(ce_vals)

                    valid_accuracy_vals = []
                    ce_vals = []
                    for i in range(valid_num_examples // batch_size):
                        batch_xs = valid_images[i*batch_size:(i+1)*batch_size, :]
                        batch_ys = valid_labels[i*batch_size:(i+1)*batch_size, :]       
                        valid_ce, accuracy = session.run([red_mean, accuracy_op], {x: batch_xs, y: batch_ys})
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
                        best_model = saver.save(session, os.path.join(save_dir + "models/", save_prefix + "homework_2-0_" + arch + '_' + str(learning_rate) + '_' + str(batch_size) + '_' + str(early_stop) +  "_" + str(model_no)))
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
                    test_ce, conf_matrix, test_accuracy = session.run([red_mean, confusion_matrix_op, accuracy_op], {x: batch_xs, y: batch_ys})
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