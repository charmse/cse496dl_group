import os
import util
import model
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imread
from PIL import Image
from random import randrange
#%matplotlib inline

#cleverhans
from cleverhans.attacks import FastGradientMethod

#inception
from tensorflow.contrib.slim.nets import inception

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_integer('early_stop', 20, '')
flags.DEFINE_string('db', 'emodb', '')
flags.DEFINE_integer('epochs', 100, '')
flags.DEFINE_float('reg_coeff', 0.001, '')
flags.DEFINE_float('split', 0.90, '')
flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
flags.DEFINE_string('checkpoint_path', 'nips-2017-adversarial-learning-development-set/inception_v3.ckpt', 'Path to checkpoint for inception network.')
flags.DEFINE_string('input_dir', 'nips-2017-adversarial-learning-development-set/images/', 'Input directory with images.')
flags.DEFINE_string('output_dir', '', 'Output directory with images.')
flags.DEFINE_float('max_epsilon', 4.0, 'Maximum size of adversarial perturbation.')
flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
flags.DEFINE_float('eps', 2.0 * 16.0 / 255.0, '')
flags.DEFINE_integer('num_classese', 1001, '')
FLAGS = flags.FLAGS

def main(argv):

    # Set arguments:  Save_Dir Structure Learning_Rate Earling_Stoping Batch_Size Data_Dir    
    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    learning_rate = FLAGS.lr
    early_stop = FLAGS.early_stop
    batch_size = FLAGS.batch_size
    reg_coeff = FLAGS.reg_coeff
    split = FLAGS.split
    master = FLAGS.master
    checkpoint_path = FLAGS.checkpoint_path
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir
    image_width = FLAGS.image_width
    image_height = FLAGS.eps
    num_classes = FLAGS.num_classes
    eps = FLAGS.eps
    batch_shape = [batch_size, image_height, image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    image_labels = pd.read_csv("nips-2017-adversarial-learning-development-set/images.csv")
    predictions = []


class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs

if __name__ == "__main__":
    tf.app.run()