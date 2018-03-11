import tensorflow as tf
import numpy as np

def psnr(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    psnr = 20*np.log10(255) - 10*np.log10(err)
    return psnr

def upscale_block(x, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, 1, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    print(n,h,w,c)
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, padding='same')

def kl_divergence(p, q):
    """ inputs must be in (0, 1) """
    return p * tf.log(p/q) + (1-p) * tf.log((1-p)/(1-q))

def autoencoder_network(x, code_size=50):
    encoder_16 = downscale_block(x)
    encoder_8 = downscale_block(encoder_16)
    flatten_dim = np.prod(encoder_8.get_shape().as_list()[1:])
    flat = tf.reshape(encoder_8, [-1, flatten_dim])
    code = tf.layers.dense(flat, code_size, activation=tf.nn.relu)
    hidden_decoder = tf.layers.dense(code, 192, activation=tf.nn.elu)
    decoder_8 = tf.reshape(hidden_decoder, [-1, 8, 8, 3])
    decoder_16 = upscale_block(decoder_8)
    output = upscale_block(decoder_16)
    return code, output

def main(argv):
    cifar_dir = '/work/cse496dl/vsunkara/03/homework/03/data'
    cifar100_train_data = np.load(cifar_dir + 'train_x.npy')
    cifar100_train_labels = np.load(cifar_dir + 'train_y.npy')
    cifar100_test_data = np.load(cifar_dir + 'test_x.npy')
    cifar100_test_labels = np.load(cifar_dir + 'test_y.npy')
    # set hyperparameters
    sparsity_weight = 5e-3
    code_size = 100

    # define graph
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    code, outputs = autoencoder_network(x, code_size)

    # calculate loss
    sparsity_loss = tf.norm(code, ord=1, axis=1)
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - x)) # MSE
    total_loss = reconstruction_loss + sparsity_weight * sparsity_loss

    # setup optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(total_loss)

    # train for epochs and visualize
    batch_size = 16

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(100):
            for i in range(cifar10_train_data.shape[0] // batch_size):
                batch_xs = cifar10_train_data[i*batch_size:(i+1)*batch_size, :]
                session.run(train_op, {x: batch_xs})

        #Run a test
        x_out, code_out, output_out = session.run([x, code, outputs], {x: np.expand_dims(cifar10_test_data[1], axis=0)})
        print(psnr(cifar10_test_data[1],output_out)

if __name__ == "__main__":
    tf.app.run()

