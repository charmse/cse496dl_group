import itertools as itr
import tensorflow as tf 
import numpy as np
import util

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

    struct = model_name.split("_")
    struct = struct[3].split("|")
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

    with tf.Session() as session:
        saver = tf.train.import_meta_graph(model_name + '.meta')
        saver.restore(session,model_name)
        graph = session.graph
        x = graph.get_tensor_by_name('input_placeholder:0')
        conv_out = graph.get_tensor_by_name('homework_02_model/conv_block_1/max_pooling2d_'+str(conv_block_size)+'/MaxPool:0')
        flat_shape = int(np.prod(np.shape(conv_out)[1:]))
        conv_out = tf.reshape(tf.stop_gradient(conv_out),[-1, flat_shape])
        block_output = util.dense_block(conv_out, neurons_or_filters, activs, regs, 1)
        output = tf.layers.dense(block_output, 7, name = 'output_layer')
    tf.identity(output, name='output2')
    return x, output
    # with tf.name_scope('optimizer') as scope:
    #     cross_entropy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
    #     red_mean1 = tf.reduce_mean(cross_entropy1)
    #     Aoptimizer = tf.train.AdamOptimizer()
    #     train_op = Aoptimizer.minimize(cross_entropy1)
    #     confusion_matrix_op1 = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=7)
    #     accuracy_op1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output2, axis=1), tf.argmax(y, axis=1)) , tf.float32))
    # optimizer_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"optimizer")
    # dense_vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"dense_1")
    # drop_vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"drop_1")
    # dense_vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"dense_2")
    # drop_vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"drop_2")
    # dense_vars_3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"dense_3")
    # drop_vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"drop_3")
    # output_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"output2")
    # session.run(tf.variables_initializer(optimizer_vars + dense_vars_1+drop_vars_1+dense_vars_2+drop_vars_2+dense_vars_3+drop_vars_3+output_vars, name = 'init'))