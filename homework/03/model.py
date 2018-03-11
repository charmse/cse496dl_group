import itertools as itr
import tensorflow as tf 
import numpy as np
import util

def upscale_block(x, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, 3, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, padding='same')

def make(x,struct):
    """
    D/C/S:n1,n2,...;;a1,a2,..;t1,t2,..;r1,r2,..;k1,k2,..;2,4.3,,..| 
    """
    struct = struct.split("|")
    block_output = x
    conv_block_num = 1
    sep_conv_block_num = 1
    dense_block_num = 1
    with tf.name_scope('homework_02_model') as scope:
        for s in struct:
            s_split = s.split(":")
            block_type = s_split[0]
            arch_and_params = s_split[1].split(";")
            neurons_or_filters = list(map(int, arch_and_params[0].split(",")))
            activs = util.get_activators(arch_and_params[1].split(","))
            types = arch_and_params[2].split(",")
            reg_vals = arch_and_params[3].split(",")
            regs = util.get_regularizers(types, reg_vals)
            if len(arch_and_params) > 4:
                kernels = list(map(int, arch_and_params[4].split(",")))
                pools = util.get_pools(arch_and_params[5].split(","))

            if block_type == "D":
                if len(np.shape(block_output)) > 2:
                    flat_shape = int(np.prod(np.shape(block_output)[1:]))
                    block_output = tf.reshape(block_output, [-1, flat_shape])
                block_output = util.dense_block(block_output, neurons_or_filters, activs, regs, dense_block_num)
                dense_block_num += 1
            elif block_type == "C1":
                if len(np.shape(block_output)) > 2:
                    flat_shape = int(np.prod(np.shape(block_output)[1:]))
                    block_output = tf.reshape(block_output, [-1, flat_shape])
                block_output = util.conv1d_block(block_output, neurons_or_filters, kernels, activs, regs, conv_block_num)
                conv_block_num += 1
            elif block_type == "C2":
                if len(np.shape(block_output)) == 2:
                    shape = int(np.sqrt(int(np.shape(block_output)[1])))
                    block_output = tf.reshape(block_output, [-1, shape, shape, 1])
                block_output = util.conv2d_block(block_output, neurons_or_filters, kernels, activs, regs, pools, conv_block_num)
                conv_block_num += 1
            elif block_type == "C3":
                if len(np.shape(block_output)) == 2:
                    shape = int(np.cbrt(int(np.shape(block_output)[1])))
                    block_output = tf.reshape(block_output, [-1, shape, shape, shape, 1])
                elif len(np.shape(block_output)) == 4:
                    shape = int(np.cbrt(np.prod(np.shape(block_output)[1:])))
                    block_output = tf.reshape(block_output, [-1, shape, shape, shape, 1])
                block_output = util.conv3d_block(block_output, neurons_or_filters, kernels, activs, regs, conv_block_num)
                conv_block_num += 1
            else:
                block_output = util.sep_conv2d_block(block_output, neurons_or_filters, kernels, activs, regs, sep_conv_block_num)
                sep_conv_block_num += 1

        if len(np.shape(block_output)) > 2:
            flat_shape = int(np.prod(np.shape(block_output)[1:]))
            block_output = tf.reshape(block_output, [-1, flat_shape])
        output = tf.layers.dense(block_output, 7, name = 'output_layer')
    tf.identity(output, name='output')
    return output

def transfer(model_name):

    struct = model_name.split("_")[3]
    arch = struct
    struct = struct.split("|")
    conv_block = struct[0].split(":")[1].split(";")
    dense_block = struct[1].split(":")[1].split(";")
    neurons_or_filters = list(map(int, dense_block[0].split(",")))
    activs = util.get_activators(dense_block[1].split(","))
    types = dense_block[2].split(",")
    reg_vals = dense_block[3].split(",")
    regs = util.get_regularizers(types, reg_vals)
    # block_output = x
    # conv_block_num = 1
    # sep_conv_block_num = 1
    # dense_block_num = 1

    conv_block_size = len(conv_block[0].split(","))

    session = tf.Session()
    saver = tf.train.import_meta_graph(model_name + '.meta')
    saver.restore(session,model_name)
    graph = session.graph
    x = graph.get_tensor_by_name('input_placeholder:0')
    conv_out = graph.get_tensor_by_name('homework_02_model/conv_block_1/max_pooling2d_'+str(conv_block_size)+'/MaxPool:0')
    flat_shape = int(np.prod(np.shape(conv_out)[1:]))
    conv_out_no_gradient = tf.reshape(tf.stop_gradient(conv_out),[-1, flat_shape])
    block_output = util.dense_block(conv_out_no_gradient, neurons_or_filters, activs, regs, 'new')
    output = tf.layers.dense(block_output, 7, name = 'output2')
    return x, output, arch

def autoencoder_network(x, code_size=100):
    encoder_16 = downscale_block(x)
    encoder_8 = downscale_block(encoder_16)
    flatten_dim = np.prod(encoder_8.get_shape().as_list()[1:])
    flat = tf.reshape(encoder_8, [-1, flatten_dim])
    code_en = tf.layers.dense(flat, 96, activation=tf.nn.relu, name='encoder_output')
    code = tf.layers.dense(code_en, code_size, activation=tf.nn.relu, name= 'code')
    code_de = tf.layers.dense(code, 96, activation=tf.nn.relu, name='decoder_input')
    hidden_decoder = tf.layers.dense(code_de, 192, activation=tf.nn.elu)
    decoder_8 = tf.reshape(hidden_decoder, [-1, 8, 8, 3])
    decoder_16 = upscale_block(decoder_8)
    outputs = upscale_block(decoder_16)
    tf.identity(outputs, name='decoder_output')
    return code, outputs