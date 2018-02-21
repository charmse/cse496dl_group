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