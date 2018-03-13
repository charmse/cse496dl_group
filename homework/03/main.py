import itertools as itr
import tensorflow as tf 
import numpy as np
import os
import sys
import model
import util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'directory where MNIST is located')
flags.DEFINE_string('train_dir', '', 'directory where training data is location')
flags.DEFINE_string('test_dir', '', '')
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
flags.DEFINE_string('ae', 'default', '')
flags.DEFINE_integer('code_size', 100, '')
flags.DEFINE_float('sparsity_weight', 5e-3, '')
flags.DEFINE_boolean('validate', False, '')
FLAGS = flags.FLAGS

def main(argv):

    # Set arguments:  Save_Dir Structure Learning_Rate Earling_Stoping Batch_Size Data_Dir    
    arch = FLAGS.arch
    save_dir = FLAGS.save_dir
    data_dir = FLAGS.data_dir
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
    train_dir = FLAGS.train_dir
    test_dir = FLAGS.test_dir
    if not bool(test_dir) :
        test_dir = data_dir
        train_dir = data_dir

    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='encoder_input')
    code, outputs, ae_name = model.autoencoder_network(x, code_size=code_size, model=ae)
    tf.identity(code, 'encoder_output')
    arch = 'AE'
    

    # # load training data
    train_images = np.load(train_dir + 'x_train.npy')
    train_labels = np.load(train_dir + 'y_train.npy')
    # load testing data
    test_images = np.load(test_dir + 'x_test.npy')
    test_labels = np.load(test_dir + 'y_test.npy')

    # split into train and validate
    if validate:
        train_images, valid_images, train_labels, valid_labels = util.split_data(train_images, train_labels, split)
        valid_num_examples = valid_images.shape[0]
    train_num_examples = train_images.shape[0]
    test_num_examples = test_images.shape[0]

    # define classification loss
    y = tf.placeholder(tf.float32, [None, 100], name='label')

    # calculate loss
    sparsity_loss = tf.norm(code, ord=1, axis=1)
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - x)) # MSE
    total_loss = reconstruction_loss + sparsity_weight * sparsity_loss

    # setup optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

if __name__ == "__main__":
    tf.app.run()