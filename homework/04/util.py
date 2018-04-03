import itertools as itr
import tensorflow as tf 
import numpy as np
import sys
import collections
import os

EPS = 1e-10
Py3 = sys.version_info[0]

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
  return train_data, valid_data, test_data, vocabulary, reversed_dictionary

def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y

class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)

def split_data(data, labels, proportion):
    """                                                                                                                                                                                                                                                                                                 
    """
    size = data.shape[0]
    np.random.seed(42)
    indicies = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[indicies[:split_idx]], data[indicies[split_idx:]], labels[indicies[:split_idx]], labels[indicies[split_idx:]]

def transform_labels(labels):
    new_labels = []
    for l in labels:
        t_label = np.zeros(10)
        t_label[(int(l)-1)] = 1.
        new_labels.append(t_label)
    return np.array(new_labels)

def onehot(data):
    data_final = np.zeros((data.shape[0],10))
    data_final[np.arange(data.shape[0]),data.astype(int)]=1
    return data_final

def extend(x, n):
    for i in range(len(x),n):
        x.append(x[len(x)-1])
    return x

def upscale_block(x, scale=2):
    """ conv2d_transpose """
    return tf.layers.conv2d_transpose(x, 3, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, np.floor(c * 1.25), 3, strides=scale, padding='same')

def kl_divergence(p, q):
    """ inputs must be in (0, 1) """
    return p * tf.log(p/q) + (1-p) * tf.log((1-p)/(1-q))



def psnr(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) + EPS
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    psnr = 20*np.log10(255) - 10*np.log10(err)
    return psnr

def gaussian_encoder(inputs, latent_size):
    """inputs should be a tensor of images whose height and width are multiples of 4"""
    x = conv_block(inputs, 8, downscale=2)
    x = conv_block(x, 16, downscale=2)
    mean = tf.layers.dense(x, latent_size)
    log_scale = tf.layers.dense(x, latent_size)
    return mean, log_scale

def gaussian_sample(mean, log_scale):
    # noise is zero centered and std. dev. 1
    gaussian_noise = tf.random_normal(shape=tf.shape(mean))
    return mean + (tf.exp(log_scale) * gaussian_noise)

def upscale_block(x, scale=2, name="upscale_block"):
     """[Sub-Pixel Convolution](https://arxiv.org/abs/1609.05158) """
     n, w, h, c = x.get_shape().as_list()
     x = tf.layers.conv2d(x, c * scale ** 2, (3, 3), activation=tf.nn.relu, padding='same', name=name)
     output = tf.depth_to_space(x, scale)
     return output

def decoder(inputs, output_shape):
     """output_shape should be a length 3 iterable of ints"""
     h, w, c = output_shape
     initial_shape = [h // 4, w // 4, c]
     initial_size = reduce(mul, initial_shape)
     x = tf.layers.dense(inputs, initial_size // 32, name="decoder_dense")
     x = tf.reshape(x, [-1] + initial_shape)
     x = upscale_block(x, name="upscale_1")
     return upscale_block(x, name="upscale_2")

def std_gaussian_KL_divergence(mu, log_sigma):
    """Analytic KL distance between N(mu, e^log_sigma) and N(0, 1)"""
    sigma = tf.exp(log_sigma)
    return -0.5 * tf.reduce_sum(
        1 + tf.log(tf.square(sigma)) - tf.square(mu) - tf.square(sigma), 1)

def flatten(inputs):
    """
    Flattens a tensor along all non-batch dimensions.
    This is correctly a NOP if the input is already flat.
    """
    if len(shape(inputs)) == 2:
        return inputs
    else:
        size = inputs.get_shape().as_list()[1:]
        return tf.reshape(inputs, [-1, size])
